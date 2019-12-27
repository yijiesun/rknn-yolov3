#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <fstream>
#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <signal.h>

#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include "src/config.h"
#include "src/v4l2/v4l2.h"  
#include "src/screen/screen.h"

#include <unistd.h>
#include <sys/syscall.h>

using namespace std;
using namespace cv;

#define MAX_QUEUE_SIZE 20
#define __TIC__(tag) timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        int ____##tag##_total_time=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        LOGD("%s :%f ms ",#tag, ____##tag##_total_time/1000.0);
typedef pair<int, Mat> imagePair;
class paircompbig {
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) return n1.first < n2.first;
        return n1.first < n2.first;
    }
};
class paircompless {
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) return n1.first > n2.first;
        return n1.first > n2.first;
    }
};

cv::VideoCapture capture;
VideoWriter outputVideo;
V4L2 v4l2_;
CONFIG config;
unsigned int * pfb;
SCREEN screen_;
int IMG_WID;
int IMG_HGT;
cv::Mat frame;

int fps_usleep;
bool use_camera_not_video,save_video,quit;

struct timeval show_img_time;
priority_queue<imagePair, vector<imagePair>, paircompless>queueFrameIn; 
priority_queue<imagePair, vector<imagePair>, paircompless> queueShow;
pthread_mutex_t  mutex_show;
pthread_mutex_t  mutex_frameIn;
pthread_mutex_t  mutex_quit;

inline int set_cpu(int i);
void my_handler(int s);
void *v4l2_thread(void *threadarg);
void *npu_thread(void *threadarg);
void *screen_thread(void *threadarg);


int idxInputImage = 0;  // image index of input video
int idxShowImage = 0;   // next frame index to be display
bool bReading = true;   // flag of input

int multi_npu_process_initialized[2] = {0, 0};

int demo_done=0;

Scalar colorArray[10]={
	Scalar(139,0,0,255),
	Scalar(139,0,139,255),
	Scalar(0,0,139,255),
	Scalar(0,100,0,255),
	Scalar(139,139,0,255),
	Scalar(209,206,0,255),
	Scalar(0,127,255,255),
	Scalar(139,61,72,255),
	Scalar(0,255,0,255),
	Scalar(255,0,0,255),
};
typedef struct{
	float x,y,w,h;
}box;
typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float objectness;
    int sort_class;
} detection;


static string labels[80]={"person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light","fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant","bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet ","tvmonitor","laptop","mouse","remote ","keyboard ","cell phone","microwave ","oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "};
detection* dets = 0;
rknn_context ctx = 0;
rknn_tensor_attr outputs_attr[3];
static int GRID0=19;
static int GRID1=38;
static int GRID2=76;
static int nclasses=80;
static int nyolo=3; //n yolo layers;
static int nanchor=3; //n anchor per yolo layer

static int nboxes_0=GRID0*GRID0*nanchor;
static int nboxes_1=GRID1*GRID1*nanchor;
static int nboxes_2=GRID2*GRID2*nanchor;
static int nboxes_total=nboxes_0+nboxes_1+nboxes_2;

float OBJ_THRESH=0.6;
float DRAW_CLASS_THRESH=0.6;
float NMS_THRESH=0.4;  //darknet demo nms=0.4

void free_detections(detection *dets, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(dets[i].prob);
    }
    free(dets);
}

double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int netw, int neth, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / netw;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / neth;
    return b;
}

void get_network_boxes(float *predictions, int netw,int neth,int GRID,int* masks, float* anchors, int box_off, detection* dets)
{
	int lw=GRID;
	int lh=GRID;
	int nboxes=GRID*GRID*nanchor;
	int LISTSIZE=1+4+nclasses;
	//darkent output排列格式: box顺序为先grid再anchor
	//1个anchor: 7*7*x+7*7*y+7*7*w+7*7*w+7*7*obj+7*7*classes1+7*7*classes2..+7*7*classes80,共3个anchor
	//x和y(先做,或者放在后面转换到实际坐标这一步中也行),以及obj和classes做logisic
	//xy做logistic
	for(int n=0;n<nanchor;n++){
		int index=n*lw*lh*LISTSIZE;
		int index_end=index+2*lw*lh;
		for(int i=index;i<index_end;i++)
			predictions[i]=1./(1.+exp(-predictions[i]));			
	}
	//类别和obj做logistic
	for(int n=0;n<nanchor;n++){
		int index=n*lw*lh*LISTSIZE+4*lw*lh;
		int index_end=index+(1+nclasses)*lw*lh;
		for(int i=index;i<index_end;i++){
			predictions[i]=1./(1.+exp(-predictions[i]));		
		}
	}
	//dets将outpus重排列,dets[i]为第i个框,box顺序为先anchor再grid

	int count=box_off;
	for(int i=0;i<lw*lh;i++){
		int row=i/lw;
		int col=i%lw;
		for(int n=0;n<nanchor;n++){
			int box_loc=n*lw*lh+i;  
			int box_index=n*lw*lh*LISTSIZE+i;            //box的x索引,ywh索引只要依次加上lw*lh
			int obj_index=box_index+4*lw*lh;
			float objectness=predictions[obj_index];
			if(objectness<OBJ_THRESH) continue;
			dets[count].objectness=objectness;
			dets[count].classes=nclasses;
			dets[count].bbox=get_yolo_box(predictions,anchors,masks[n],box_index,col,row,lw,lh,netw,neth,lw*lh);
			for(int j=0;j<1;j++){
				int class_index=box_index+(5+j)*lw*lh;
				float prob=objectness*predictions[class_index];
				dets[count].prob[j]=prob;
				if(prob*100>=OBJ_THRESH && j==0)
					cout<<labels[j]<<"-- "<<prob*100<<endl;
			}
			++count;
		}
	}
	//cout<<"count: "<<count-box_off<<"\n";
    //return dets;
}

void outputs_transform(rknn_output rknn_outputs[], int net_width, int net_height, detection* dets){
	float* output_0=(float*)rknn_outputs[0].buf;
	float* output_1=(float*)rknn_outputs[1].buf;
	float* output_2=(float*)rknn_outputs[2].buf;
	int masks_0[3] = {6, 7, 8};
    	int masks_1[3] = {3, 4, 5};
	int masks_2[3] = {0, 1, 2};
	//float anchors[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
	float anchors[18] = {10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};
	//输出xywh均在0-1范围内
	get_network_boxes(output_0,net_width,net_height,GRID0,masks_0,anchors,0,dets);
	get_network_boxes(output_1,net_width,net_height,GRID1,masks_1,anchors,nboxes_0,dets);
	get_network_boxes(output_2,net_width,net_height,GRID2,masks_2,anchors,nboxes_0+nboxes_1,dets);
	//return dets;
}

float overlap(float x1,float w1,float x2,float w2){
	float l1=x1-w1/2;
	float l2=x2-w2/2;
	float left=l1>l2? l1:l2;
	float r1=x1+w1/2;
	float r2=x2+w2/2;
	float right=r1<r2? r1:r2;
	return right-left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

int nms_comparator(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if(b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}
int do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total-1;
    for(i = 0; i <= k; ++i){
        if(dets[i].objectness == 0){
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k+1;
	//cout<<"total after OBJ_THRESH: "<<total<<"\n";

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator);
        for(i = 0; i < total; ++i){
            if(dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for(j = i+1; j < total; ++j){
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
	return total;
}


int draw_image(cv::Mat img,detection* dets,int total,float thresh)
{
	//::cvtColor(img, img, cv::COLOR_RGB2BGR);
	for(int i=0;i<total;i++){
		char labelstr[4096]={0};
		int class_=-1;
		int topclass=-1;
		float topclass_score=0;
		if(dets[i].objectness==0) continue;
		for(int j=0;j<nclasses;j++){
			if(dets[i].prob[j]>thresh){
				if(topclass_score<dets[i].prob[j]){
					topclass_score=dets[i].prob[j];
					topclass=j;
				}
				if(class_<0){
					strcat(labelstr,labels[j].data());
					class_=j;
				}
				else{
					strcat(labelstr,",");
					strcat(labelstr,labels[j].data());
				}
				// printf("%s: %.02f%%\n",labels[j].data(),dets[i].prob[j]*100);
			}
		}
		//如果class>0说明框中有物体,需画框
		if(class_>=0){
			box b=dets[i].bbox;
			int x1 =(b.x-b.w/2.)*img.cols;
			int x2=(b.x+b.w/2.)*img.cols;
			int y1=(b.y-b.h/2.)*img.rows;
			int y2=(b.y+b.h/2.)*img.rows;

            if(x1  < 0) x1  = 0;
            if(x2> img.cols-1) x2 = img.cols-1;
            if(y1 < 0) y1 = 0;
            if(x2 > img.rows-1) x2 = img.rows-1;
			// std::cout << labels[topclass] << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

            rectangle(img, Point(x1, y1), Point(x2, y2), colorArray[class_%10], 3);
            putText(img, labelstr, Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
            }
		}
	return 0;
}

int main(const int argc, const char** argv) 
{
	IMG_HGT = 480;
    IMG_WID = 640;
  	std::string in_video_file;
    std::string out_video_file;
    config.get_param_mssd_video_knn(in_video_file,out_video_file);
    LOGD("save  video:   %s",out_video_file.c_str());
    std::string dev_num;
    config.get_param_mms_V4L2(dev_num);
    LOGD("open  %s",dev_num.c_str());
    config.get_use_camera_or_video(use_camera_not_video);
    LOGD("use_camera_not_video  %d",use_camera_not_video);
    config.get_save_video(save_video);
    LOGD("save_video  %d",save_video);
    config.get_fps_sleep(fps_usleep);
    LOGD("fps_usleep  %d",fps_usleep);

    frame.create(IMG_HGT,IMG_WID,CV_8UC3);
    frame = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);

    screen_.init((char *)"/dev/fb0",640,480);
    pfb = (unsigned int *)malloc(screen_.finfo.smem_len);

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

	if(use_camera_not_video)
    {
        capture.open(in_video_file.c_str());
        capture.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));
    }
    else
    {
        v4l2_.init(dev_num.c_str(),640,480);
        v4l2_.open_device();
        v4l2_.init_device();
        v4l2_.start_capturing();
    }

    if(save_video)
    {
        Size sWH = Size( IMG_WID,IMG_HGT);
        outputVideo.open(out_video_file.c_str(), CV_FOURCC('M', 'J', 'P', 'G'), 25, sWH,true);
        if(!outputVideo.isOpened())
            LOGD("save video failed!");
    }
    
	const char *model_path = "../../yolov3_608x608.rknn";
	dets =(detection*) calloc(nboxes_total,sizeof(detection));
	for(int i = 0; i < nboxes_total; ++i)
		dets[i].prob = (float*) calloc(nclasses,sizeof(float));
	// Load model
	FILE *fp = fopen(model_path, "rb");
    	if(fp == NULL) {
        	printf("fopen %s fail!\n", model_path);
    	}
    	fseek(fp, 0, SEEK_END);   //fp指向end,fseek(FILE *stream, long offset, int fromwhere);
    	int model_len = ftell(fp);   //相对文件首偏移
    	void *model = malloc(model_len);
    	fseek(fp, 0, SEEK_SET);   //SEEK_SET为文件头
    	if(model_len != fread(model, 1, model_len, fp)) {
        	printf("fread %s fail!\n", model_path);
        	free(model);
    	}
	
	//init
	
	int ret = rknn_init(&ctx,model,model_len,RKNN_FLAG_PRIOR_MEDIUM);
	if(ret < 0) {
        	printf("rknn_init fail! ret=%d\n", ret);
    	}


	//rknn outputs_attr
	outputs_attr[0].index = 0;
	ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
	if(ret < 0) {
	    printf("rknn_query fail! ret=%d\n", ret);
	}


	int NPU0=0; 
	int NPU1=1; 
	pthread_t threads_npu;   
    pthread_t threads_v4l2;
    pthread_t threads_screen;
    pthread_create(&threads_npu, NULL, npu_thread, NULL);
    pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);
    pthread_create(&threads_screen, NULL,screen_thread, NULL);
    pthread_join(threads_npu,NULL);
    pthread_join(threads_v4l2,NULL);
    pthread_join(threads_screen,NULL);

    if(save_video)
        outputVideo.release();
            // release handle

    if(use_camera_not_video)
    {
        capture.release();
    }
    else
    {
        v4l2_.stop_capturing();
        v4l2_.uninit_device();
        v4l2_.close_device();
    }
    LOGD("exit success!");
    return 0;
}

void *v4l2_thread(void *threadarg)
{
    set_cpu(3);
    while(1)
    {
        pthread_mutex_lock(&mutex_quit);
        if(quit)
        {
            pthread_mutex_unlock(&mutex_quit);
            break;
        }
        else
            pthread_mutex_unlock(&mutex_quit);

        __TIC__(CAMERA);
        if(use_camera_not_video)
        {
            if(!capture.read(frame))
            {
                pthread_mutex_lock(&mutex_quit);
                quit = true;
                pthread_mutex_unlock(&mutex_quit);
            }
        }
        else
        {
            v4l2_.read_frame(frame);
        }

        int time_ = config.getTimesInt();
        imagePair pframe(time_,frame);
        pthread_mutex_lock(&mutex_frameIn);
        if(queueFrameIn.size()<MAX_QUEUE_SIZE)
            queueFrameIn.push(pframe);
        pthread_mutex_unlock(&mutex_frameIn);
        usleep(fps_usleep); 
        __TOC__(CAMERA);
        
    }
    pthread_exit(NULL);
}

void *screen_thread(void *threadarg)
{
    set_cpu(4);

    while(1)
    {    
        pthread_mutex_lock(&mutex_quit);
        if(quit)
        {
            pthread_mutex_unlock(&mutex_quit);
            break;
        }
        else
            pthread_mutex_unlock(&mutex_quit);

        pthread_mutex_lock(&mutex_show);
         if(!queueShow.empty())
        {
            __TIC__(SHOW);
            pthread_mutex_unlock(&mutex_show);
            std::string fps_str;
            struct timeval t1;
            gettimeofday(&t1, NULL);
            float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (show_img_time.tv_sec * 1000000 + show_img_time.tv_usec)) / 1000.0;
            int fps = (int)(1000.0/mytime);
            char buffer[32];
            std::sprintf(buffer, "%d",fps);
            string fps_s = buffer;
            fps_str = "fps:"+fps_s;

            pthread_mutex_lock(&mutex_show);
            Mat show_img = queueShow.top().second;
            queueShow.pop();
            pthread_mutex_unlock(&mutex_show);

            Point siteNo;
            siteNo.x = 25;
            siteNo.y = 25;
            putText( show_img, fps_str, siteNo, 2,1,Scalar( 255, 0, 0 ), 4);
            v4l2_. mat_to_argb(show_img.data,pfb,IMG_WID,IMG_HGT,screen_.vinfo.xres_virtual,0,0);
            memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);

            if(save_video)
            {
                outputVideo.write(show_img);
            }
               
            show_img_time.tv_sec=0;
            show_img_time.tv_usec=0;
            gettimeofday(&show_img_time, NULL);
            __TOC__(SHOW);
        }
        else
        {
            pthread_mutex_unlock(&mutex_show);
        }
        usleep(10000); 
    }
    pthread_exit(NULL);
}

void *npu_thread(void *threadarg)
{
    set_cpu(5);

    const int net_width=608;
	const int net_height=608;
	const int img_channels=3;
	rknn_input inputs[1];
	//rknn inputs
	inputs[0].index = 0;
	inputs[0].size = net_width * net_height * img_channels;
	inputs[0].pass_through = false;         //需要type和fmt
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
				//rknn outputs
	rknn_output outputs[3];
	outputs[0].want_float = true;
	outputs[0].is_prealloc = false;
	outputs[1].want_float = true;
	outputs[1].is_prealloc = false;
	outputs[2].want_float = true;
	outputs[2].is_prealloc = false;
    while(1)
    {
        pthread_mutex_lock(&mutex_quit);
        if(quit)
        {
            pthread_mutex_unlock(&mutex_quit);
            break;
        }
        else
            pthread_mutex_unlock(&mutex_quit);

        Mat img_roi;
        pthread_mutex_lock(&mutex_frameIn);
        if(!queueFrameIn.empty())
        {
            img_roi = queueFrameIn.top().second;
            queueFrameIn.pop();
        }
        else
        {
            pthread_mutex_unlock(&mutex_frameIn);
            continue;
        }
        pthread_mutex_unlock(&mutex_frameIn);

        __TIC__(NPU);
		//==================================================================================//
		// YOLO Process
		int nboxes_left = 0;
		cv::Mat resimg;
		cout<<"img_roi "<<img_roi.size()<<endl;
		cv::resize(img_roi, resimg, cv::Size(net_width, net_height), (0, 0), (0, 0), cv::INTER_LINEAR);
		inputs[0].buf = resimg.data;
		int ret = rknn_inputs_set(ctx, 1, inputs);
		if (ret < 0) {
			printf("rknn_input_set fail! ret=%d\n", ret);
			rknn_outputs_release(ctx, 3, outputs);
			continue;
		}
		
	    	ret = rknn_run(ctx, nullptr);
	    	if (ret < 0) {
	        	printf("rknn_run fail! ret=%d\n", ret);
				continue;
	    	}
		
		ret = rknn_outputs_get(ctx, 3, outputs, NULL);
		if (ret < 0) {
			printf("rknn_outputs_get fail! ret=%d\n", ret);
			rknn_outputs_release(ctx, 3, outputs);
			continue;
		}
		for(int i = 0; i < nboxes_total; ++i)
			dets[i].objectness = 0;
		
		outputs_transform(outputs, net_width, net_height, dets);

		nboxes_left=do_nms_sort(dets, nboxes_total, nclasses, NMS_THRESH);
		draw_image(img_roi, dets, nboxes_left, DRAW_CLASS_THRESH);

		rknn_outputs_release(ctx, 3, outputs);

        Mat show_img(IMG_HGT,IMG_WID,CV_8UC3);
		show_img = img_roi.clone();
        int time_ = config.getTimesInt();
        imagePair pframe(time_,show_img);
        pthread_mutex_lock(&mutex_show);
        queueShow.push(pframe);
        pthread_mutex_unlock(&mutex_show);

        __TOC__(NPU);

        // sleep(0.04); 
    }
    pthread_exit(NULL);
}

void my_handler(int s)
{
            pthread_mutex_lock(&mutex_quit);
            quit=true;
            pthread_mutex_unlock(&mutex_quit);
            LOGD("Caught signal  %d  quit= %d",s,quit);
}

inline int set_cpu(int i)  
{  
    cpu_set_t mask;  
    CPU_ZERO(&mask);  
  
    CPU_SET(i,&mask);  

    if(-1 == pthread_setaffinity_np(pthread_self() ,sizeof(mask),&mask))  
    {  
        fprintf(stderr, "pthread_setaffinity_np erro\n");  
        return -1;  
    }  
    return 0;  
} 
