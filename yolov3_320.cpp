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

#include "src/config.h"
#include "src/v4l2/v4l2.h"  
#include "src/screen/screen.h"
#include "rknn_api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <sys/syscall.h>

using namespace std;
using namespace cv;

#define MAX_QUEUE_SIZE 20
int idxInputImage = 0;  // image index of input video
int idxShowImage = 0;   // next frame index to be display
bool bReading = true;   // flag of input

typedef pair<int, Mat> imagePair;
class paircomp {
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
bool use_video_not_camera,save_video,quit;

mutex mtxQuit; 
mutex mtxQueueInput;               // mutex of input queue
mutex mtxQueueShow;                // mutex of display queue
queue<pair<int, Mat>> queueInput;  // input queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;  // display queue

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

static string labels[80]={"person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light","fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant","bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet ","tvmonitor","laptop","mouse","remote ","keyboard ","cell phone","microwave ","oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush "};

static int GRID0=10;
static int GRID1=20;
static int GRID2=40;
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

void my_handler(int s)
{
		mtxQuit.lock();
        quit=true;
		mtxQuit.unlock();
        LOGD("Caught signal  %d  quit= %d",s,quit);
}

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
			for(int j=0;j<nclasses;j++){
				int class_index=box_index+(5+j)*lw*lh;
				float prob=objectness*predictions[class_index];
				dets[count].prob[j]=prob;
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
				//printf("%s: %.02f%%\n",labels[j].data(),dets[i].prob[j]*100);
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
			//std::cout << labels[topclass] << "\t@ (" << x1 << ", " << y1 << ") (" << x2 << ", " << y2 << ")" << "\n";

            rectangle(img, Point(x1, y1), Point(x2, y2), colorArray[class_%10], 3);
            putText(img, labelstr, Point(x1, y1 - 12), 1, 2, Scalar(0, 255, 0, 255));
            }
		}
	return 0;
}


void run_process(int thread_id)
{
	const char *model_path = "../../yolov3_320x320.rknn";
	const int net_width=320;
	const int net_height=320;
	const int img_channels=3;
	cpu_set_t mask;
	int cpuid = 0;
	int ret = 0;

	if (thread_id == 0)
		cpuid = 4;
	else if (thread_id == 1)
		cpuid = 5;
	else
		cpuid = 0;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind NPU process(%d) to CPU %d\n", thread_id, cpuid);

	rknn_input inputs[1];
  	rknn_output outputs[3];
  	rknn_tensor_attr outputs_attr[3];
	detection* dets = 0;

	dets =(detection*) calloc(nboxes_total,sizeof(detection));
	for(int i = 0; i < nboxes_total; ++i)
		dets[i].prob = (float*) calloc(nclasses,sizeof(float));

	// Load model
	FILE *fp = fopen(model_path, "rb");
    	if(fp == NULL) {
        	printf("fopen %s fail!\n", model_path);
        	return;
    	}
    	fseek(fp, 0, SEEK_END);   //fp指向end,fseek(FILE *stream, long offset, int fromwhere);
    	int model_len = ftell(fp);   //相对文件首偏移
    	void *model = malloc(model_len);
    	fseek(fp, 0, SEEK_SET);   //SEEK_SET为文件头
    	if(model_len != fread(model, 1, model_len, fp)) {
        	printf("fread %s fail!\n", model_path);
        	free(model);
        	return;
    	}
	
	//init
	rknn_context ctx = 0;
	ret = rknn_init(&ctx,model,model_len,RKNN_FLAG_PRIOR_MEDIUM);
	if(ret < 0) {
        	printf("rknn_init fail! ret=%d\n", ret);
        	return;
    	}
	
	//rknn inputs
	inputs[0].index = 0;
	inputs[0].size = net_width * net_height * img_channels;
	inputs[0].pass_through = false;         //需要type和fmt
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].fmt = RKNN_TENSOR_NHWC;

	//rknn outputs
	outputs[0].want_float = true;
	outputs[0].is_prealloc = false;
	outputs[1].want_float = true;
	outputs[1].is_prealloc = false;
	outputs[2].want_float = true;
	outputs[2].is_prealloc = false;

	//rknn outputs_attr
	outputs_attr[0].index = 0;
	ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
	if(ret < 0) {
	    printf("rknn_query fail! ret=%d\n", ret);
	    return;
	}

	multi_npu_process_initialized[thread_id] = 1;
  	printf("The initialization of NPU Process %d has been completed.\n", thread_id);

	while (true)
	{
		mtxQuit.lock();
		if(quit)
		{
			mtxQuit.unlock();
			break;
		}
		mtxQuit.unlock();

		double start_time,end_time;

		pair<int, Mat> pairIndexImage;
		mtxQueueInput.lock();
		if (queueInput.empty()) {
			//printf("waiting queueInput .........\n", ret);
			mtxQueueInput.unlock();
			usleep(1000);
			if (bReading) {
				continue;
			} else {
				rknn_destroy(ctx);
				break;
			}
		} else {
			// Get an image from input queue
			pairIndexImage = queueInput.front();
			queueInput.pop();
			mtxQueueInput.unlock();
		}
		
		//==================================================================================//
		// YOLO Process
		int nboxes_left = 0;
		cv::Mat resimg;
		
		cv::resize(pairIndexImage.second, resimg, cv::Size(net_width, net_height), (0, 0), (0, 0), cv::INTER_LINEAR);
		//cv::cvtColor(resimg, resimg, cv::COLOR_BGR2RGB);
		
		start_time=what_time_is_it_now();
		inputs[0].buf = resimg.data;
		ret = rknn_inputs_set(ctx, 1, inputs);
		if (ret < 0) {
			printf("rknn_input_set fail! ret=%d\n", ret);
			return;
		}
		
	    	ret = rknn_run(ctx, nullptr);
	    	if (ret < 0) {
	        	printf("rknn_run fail! ret=%d\n", ret);
	        	return;
	    	}

		ret = rknn_outputs_get(ctx, 3, outputs, NULL);
		if (ret < 0) {
			printf("rknn_outputs_get fail! ret=%d\n", ret);
			return;
		}
		end_time=what_time_is_it_now();
		//cout<<"rknn use time: "<<(end_time - start_time)<<"\n";
		
		start_time = what_time_is_it_now();
		for(int i = 0; i < nboxes_total; ++i)
			dets[i].objectness = 0;
		
		outputs_transform(outputs, net_width, net_height, dets);
		end_time=what_time_is_it_now();
		//cout<<"outputs_transform use time: "<<(end_time - start_time)<<"\n";

		start_time = what_time_is_it_now();
		nboxes_left=do_nms_sort(dets, nboxes_total, nclasses, NMS_THRESH);
		end_time=what_time_is_it_now();
		//cout<<"do_nms_sort use time: "<<(end_time - start_time)<<"\n";
	
		start_time = what_time_is_it_now();
		draw_image(pairIndexImage.second, dets, nboxes_left, DRAW_CLASS_THRESH);
		end_time=what_time_is_it_now();
		//cout<<"draw_image use time: "<<(end_time - start_time)<<"\n";


		rknn_outputs_release(ctx, 3, outputs);
		mtxQueueShow.lock();
		queueShow.push(pairIndexImage);
		mtxQueueShow.unlock();
	}
}

void cameraRead() 
{
	int i = 0;
  	int initialization_finished = 1;
  	cpu_set_t mask;
  	int cpuid = 2;

  	CPU_ZERO(&mask);
  	CPU_SET(cpuid, &mask);

  	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
    		cerr << "set thread affinity failed" << endl;

  	printf("Bind CameraCapture process to CPU %d\n", cpuid); 

  	// VideoCapture camera(index);
  	// if (!camera.isOpened()) {
  	// 	cerr << "Open camera error!" << endl;
  	// 	exit(-1);
  	// }

  	while (true) {
    		initialization_finished = 1;

    		for (int i = 0; i < sizeof(multi_npu_process_initialized) / sizeof(int); i++) {
      			//cout << i << " " << multi_npu_process_initialized << endl;
	      		if (multi_npu_process_initialized[i] == 0) {
	        		initialization_finished = 0;
      			}
    		}

    		if (initialization_finished)
      			break;
    		sleep(1);
  	}

	while (true) {
		mtxQuit.lock();
		if(quit)
		{
			mtxQuit.unlock();
			break;
		}
		mtxQuit.unlock();
		
		mtxQueueInput.lock();
		if (queueInput.size() <= MAX_QUEUE_SIZE)
		{
		usleep(1000);
		v4l2_.read_frame(frame);
		if (frame.empty()) {
			cerr << "Fail to read image from camera!" << endl;
			break;
		}
		queueInput.push(make_pair(idxInputImage++, frame));
		mtxQueueInput.unlock();

		}
		else
		{
			mtxQueueInput.unlock();
		}
				
		if (bReading) {
			continue;
		} else {
			// camera.release();
			v4l2_.stop_capturing();
			v4l2_.uninit_device();
			v4l2_.close_device();
			break;
		}
	}
}

void videoRead(const char *video_name) 
{
	int i = 0;
	int initialization_finished = 1;
	int cpuid = 2;
	cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind VideoCapture process to CPU %d\n", cpuid); 

	VideoCapture video;
	if (!video.open(video_name)) {
		cout << "Fail to open " << video_name << endl;
		mtxQuit.lock();
		quit = true;
		mtxQuit.unlock();
		return;
	}

	int frame_cnt = video.get(CV_CAP_PROP_FRAME_COUNT);
	
	while (true) {
		initialization_finished = 1;
		for (int i = 0; i < sizeof(multi_npu_process_initialized) / sizeof(int); i++) {
			if (multi_npu_process_initialized[i] == 0) {
				initialization_finished = 0;
			}
		}

		if (initialization_finished)
			break;

		sleep(1);
	}

	while (true) 
	{  
		mtxQuit.lock();
		if(quit)
		{
			mtxQuit.unlock();
			break;
		}
		mtxQuit.unlock();
		usleep(1000);
		Mat img;

		if (queueInput.size() < 30) {
			if (!video.read(img)) {
				cout << "read video stream failed! or end!" << endl;
				mtxQuit.lock();
				quit = true;
				mtxQuit.unlock();
				//video.set(CV_CAP_PROP_POS_FRAMES, 0);
				break;
			}
			if(img.size().width!=IMG_WID || img.size().height!=IMG_HGT)
				cv::resize(img, img, cv::Size(IMG_WID, IMG_HGT), (0, 0), (0, 0), cv::INTER_LINEAR);
			mtxQueueInput.lock();
			queueInput.push(make_pair(idxInputImage++, img));
			mtxQueueInput.unlock();
			usleep(10);
			
		}
		if (bReading) {
			continue;
		} else {
			video.release();
			break;
		}
	}
}
void displayImage() 
{
	Mat img;
	cpu_set_t mask;
	int cpuid = 3;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind Display process to CPU %d\n", cpuid); 

	double first_time,last_time;
	int fps;
	first_time=what_time_is_it_now();
	while (true) 
	{
		mtxQuit.lock();
		if(quit)
		{
			mtxQuit.unlock();
			break;
		}
		mtxQuit.unlock();
		mtxQueueShow.lock();
        	if (queueShow.empty()) {
			mtxQueueShow.unlock();
			usleep(1000);
		} else if (idxShowImage == queueShow.top().first) {
        		Mat img = queueShow.top().second;
            		string a = to_string(fps) + "FPS";
            		cv::putText(img, a, cv::Point(15, 15), 1, 1, cv::Scalar{0, 0, 255},2);

					v4l2_. mat_to_argb(img.data,pfb,img.size().width,img.size().height,screen_.vinfo.xres_virtual,0,0);
					memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);
					if(save_video)
					{
						outputVideo.write(img);
					}

            		// cv::imshow("RK3399Pro", img);  // display image
            		idxShowImage++;
            		queueShow.pop();
            		mtxQueueShow.unlock();

            	// 	if (waitKey(1) == 27) {
				// cv::destroyAllWindows();
				// bReading = false;
				// break;
            	// 	}
			last_time=what_time_is_it_now();
			//cout<<"all use time: "<<(last_time - first_time)<<"\n";
			fps = (int)1/(last_time - first_time);
			first_time=what_time_is_it_now();
        	} else {
            		mtxQueueShow.unlock();
        	}

    	}
		if(save_video)
        	outputVideo.release();
}

int main(const int argc, const char** argv) 
{
	quit = false;
	IMG_HGT = 480;
    IMG_WID = 640;
  	std::string in_video_file;
    std::string out_video_file;
    config.get_param_mssd_video_knn(in_video_file,out_video_file);
    LOGD("save  video:   %s",out_video_file.c_str());
    std::string dev_num;
    config.get_param_mms_V4L2(dev_num);
    LOGD("open  %s",dev_num.c_str());
    config.get_use_camera_or_video(use_video_not_camera);
    LOGD("use_video_not_camera  %d",use_video_not_camera);
    config.get_save_video(save_video);
    LOGD("save_video  %d",save_video);

    frame.create(IMG_HGT,IMG_WID,CV_8UC3);
    frame = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);

	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

    screen_.init((char *)"/dev/fb0",640,480);
    pfb = (unsigned int *)malloc(screen_.finfo.smem_len);

	if(use_video_not_camera)
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

	int i, cpus = 0;
	int camera_index;
	cpu_set_t mask;
	cpu_set_t get;
	array<thread, 4> threads;

	cpus = sysconf(_SC_NPROCESSORS_CONF);
	printf("This system has %d processor(s).\n", cpus);

	if (!use_video_not_camera) {
		threads = {thread(cameraRead),
                                thread(displayImage),
                                thread(run_process, 0),
                                thread(run_process, 1)};
	} else {
		threads = {thread(videoRead, in_video_file.c_str()),
                                thread(displayImage),
                                thread(run_process, 0),
                                thread(run_process, 1)};
	} 

	for (int i = 0; i < 4; i++)
		threads[i].join();
	LOGD("quit successed!");
	return 0;
}
