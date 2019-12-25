#include "config.h"
using namespace std;
CONFIG::CONFIG()
{
	config_file = "config.txt";
	if ((file_open = fopen(config_file.c_str(), "r")) == NULL) {
	puts("Fail to open config.txt!");
	exit(0);
	}
}
CONFIG::~CONFIG()
{
	free(file_open);
}
bool CONFIG::_str_cmp(char* a, char *b)
{	
	int sum = 0;
	for (int i = 0; b[i] != '\0'; i++)
		sum++;
	char tmp[200] = {""};
	strncpy(tmp, a + 0, sum);
	for (int i = 0; a[i] != '\0'; i++)
	{
		if (a[i] == '\n')
			a[i] = (char)NULL;
	}
	return !strcmp(tmp,b);
}

void CONFIG::get_param_mssd_img(std::string &in,std::string &out)
{
	fseek( file_open, 0, SEEK_SET );
    char img_in[200];
    char img_out[200];
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
        if (_str_cmp(str, (char *)"mssd_img_in"))
			strncpy(img_in, str + 12, 200);
		else if (_str_cmp(str, (char *)"mssd_img_out"))
			strncpy(img_out, str + 13, 200);
		
    }

    in = img_in;
    out = img_out;
}
void CONFIG::get_param_mssd_video_knn(std::string &in,std::string &out)
{
	fseek( file_open, 0, SEEK_SET );
    char img_in[200];
    char img_out[200];
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
        if (_str_cmp(str, (char *)"mssd_video_knn_in"))
			strncpy(img_in, str + 18, 200);
		else if (_str_cmp(str, (char *)"mssd_video_knn_out"))
			strncpy(img_out, str + 19, 200);
		
    }

    in = img_in;
    out = img_out;

}
void CONFIG::get_param_mssd_video(std::string &in,std::string &out)
{
	fseek( file_open, 0, SEEK_SET );
    char img_in[200];
    char img_out[200];
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
        if (_str_cmp(str, (char *)"mssd_video_in"))
			strncpy(img_in, str + 14, 200);
		else if (_str_cmp(str, (char *)"mssd_video_out"))
			strncpy(img_out, str + 15, 200);
    }

    in = img_in;
    out = img_out;

}

void CONFIG::get_param_mms_cvCaptrue(int & dev)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"mssd_cvCaptrue_dev"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &dev);
		}
		
    }

}
void CONFIG::get_param_mms_V4L2(std::string &dev)
{
	fseek( file_open, 0, SEEK_SET );
     char dev_name[200];
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
        if (_str_cmp(str, (char *)"mssd_v4l2_dev"))
			strncpy(dev_name, str + 14, 200);
    }
    dev = dev_name;

}

void CONFIG::get_camera_size(int &wid,int &hgt)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"camera_width"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &wid);
		}
		else  if (_str_cmp(str, (char *)"camera_height"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &hgt);
		}
    }

}
void CONFIG::get_show_knn_box(bool &show)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	int num;
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"show_knn_box"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &num);
		}
    }
	show = (bool)num;
}
void CONFIG::get_use_camera_or_video(bool &cov)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	int num;
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"use_camera_or_video"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &num);
		}
    }
	cov = (bool)num;
}

void CONFIG::get_save_video(bool &sv)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	int num;
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"save_video"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &num);
		}
    }
	sv = (bool)num;
}
void CONFIG::get_roi_limit(bool &roi)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	int num;
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"roi_limit"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &num);
		}
    }
	roi = (bool)num;
}
void CONFIG::get_show_img(bool &show)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	int num;
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"show_img"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &num);
		}
    }
	show = (bool)num;
}
void CONFIG::get_captrue_save_data_floder(std::string &imag,std::string &video)
{
	fseek( file_open, 0, SEEK_SET );
	char in[200];
    char out[200];
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
        if (_str_cmp(str, (char *)"captrue_data_save_img_floder"))
			strncpy(in, str + 29, 200);
		else if (_str_cmp(str, (char *)"captrue_data_save_video_floder"))
			strncpy(out, str + 31, 200);
		
    }

    imag = in;
    video = out;

}
void CONFIG::get_captrue_data_save_video_mode(int &mode)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"captrue_data_save_video_mode"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &mode);
		}
		
    }
}
void CONFIG::get_move_percent(double & move)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"captrue_data_move_percent"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%lf", &move);
		}
		
    }
}
void CONFIG::get_knn_thresh(double & th)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"knn_thresh"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%lf", &th);
		}
		
    }
}
void CONFIG::get_knn_box_exist_cnt(int & cnt)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"knn_box_exist_cnt"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &cnt);
		}
		
    }
}
void CONFIG::get_fps_sleep(int & fs)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"fps_sleep"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &fs);
		}
		
    }
}
void CONFIG::get_move_buff_cnt(int & cnt)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"captrue_data_move_buff_cnt"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &cnt);
		}
		
    }
}
void CONFIG::get_captrue_data_save_img_mode(int &mode)
{
	fseek( file_open, 0, SEEK_SET );
	char str[200];
	while (fgets(str, 100, file_open) != NULL) {
       if (_str_cmp(str, (char *)"captrue_data_save_img_mode"))
		{
			const char * split = " ";
			char *p = strtok(str, split);
			p = strtok(NULL, split);
			sscanf(p, "%d", &mode);
		}
		
    }
}
void CONFIG::getTimesSecf(char *param)
{
	time_t lastTime;
	time(&lastTime);
	time_t tt;
	time(&tt);
	tm* t;
	lastTime = tt;
	tt = tt + 8 * 3600;  // transform the time zone
	t = gmtime(&tt);

	struct  timeb   stTimeb;
	ftime(&stTimeb);
	sprintf(param, "%d-%02d-%02d-%02d-%02d-%02d-%03d",
		t->tm_year + 1900,
		t->tm_mon + 1,
		t->tm_mday,
		t->tm_hour,
		t->tm_min,
		t->tm_sec,
		stTimeb.millitm);
}
void CONFIG::getTimesSec(char *param)
{
	time_t lastTime;
	time(&lastTime);
	time_t tt;
	time(&tt);
	tm* t;
	lastTime = tt;
	tt = tt + 8 * 3600;  // transform the time zone
	t = gmtime(&tt);

	struct  timeb   stTimeb;
	ftime(&stTimeb);
	sprintf(param, "%d-%02d-%02d-%02d-%02d-%02d",
		t->tm_year + 1900,
		t->tm_mon + 1,
		t->tm_mday,
		t->tm_hour,
		t->tm_min,
		t->tm_sec);
}
int CONFIG::getTimesInt()
{
	time_t lastTime;
	time(&lastTime);
	time_t tt;
	time(&tt);
	tm* t;
	lastTime = tt;
	tt = tt + 8 * 3600;  // transform the time zone
	t = gmtime(&tt);
	
	struct  timeb   stTimeb;
	ftime(&stTimeb);
	
	int ret = (int) (t->tm_sec*+stTimeb.millitm*1000);
	return ret;
}