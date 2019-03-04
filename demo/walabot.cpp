//     Created on Nov 24 23:45 2018
//
//     Author           : Shaoshu Yang
//     Email            : shaoshuyangseu@gmail.com
//     Last edit date   : Nov 24 24:00 2018
//
//South East University Automation College, 211189 Nanjing China

#include "walabot.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/timeb.h>
#include <sys\utime.h>

walabot::walabot(DEPTH_ARENA r_min, DEPTH_ARENA r_max, DEPTH_RESOL r_res, ANGULAR_ARENA phi_min, ANGULAR_ARENA phi_max,
        ANGULAR_RESOL phi_res, ANGULAR_ARENA theta_min, ANGULAR_ARENA theta_max, ANGULAR_RESOL theta_res, FILTER filter,
        THRES threshold)
 /* Default constructor and destructor of walabot class. Walabot must be intialized with a set of coefficients.
  * The constructor routine would not initialize walabot, hence it can not perform scanning immediately.
  * Args:
  *      r_min     : Minimum depth of scanning arena
  *      r_max     : Maximum depth of scanning arena
  *      r_res     : Depth spatial resolution
  *      phi_min   : Minimum phi angle
  *      phi_max   : Maximum phi angle
  *      phi_res   : Angular resolution of phi
  *      theta_min : Minimum theta angle
  *      theta_max : Maximum theta angle
  *      theta_res : Angular resolution of theta
  *      filter    : Use motion target identification filter
  *      threshold : Threshold for the detectable signals*/
{
    _r_min = r_min; _r_max = r_max; _r_res = r_res;
    _phi_min = phi_min; _phi_max = phi_max; _phi_res = phi_res;
    _theta_min = theta_min; _theta_max = theta_max; _theta_res = theta_res;
    _MTI = filter;
}

walabot::~walabot()
/* The defualt destructor would disconncect walabot automatically.*/
{
    Walabot_Disconnect();
}

void walabot::start()
/* State control routine for walabot. After applying start routine, walabot can perform scanning and
     * collecting images.*/
{
    WALABOT_RESULT _status;                                                                         // Running status
    _status = Walabot_Initialize(CONFIG_FILE_PATH); _check_status(_status);                         // Initialize
    _status = Walabot_ConnectAny(); _check_status(_status);                                         // Connect to walabot
    _status = Walabot_SetProfile(PROF_SENSOR); _check_status(_status);                              // Scanning configuration
    _status = Walabot_SetArenaR(_r_min, _r_max, _r_res); _check_status(_status);                    // Set scan depth
    _status = Walabot_SetArenaPhi(_phi_min, _phi_max, _phi_res); _check_status(_status);            // Set angular arena
    _status = Walabot_SetArenaTheta(_theta_min, _theta_max, _theta_res); _check_status(_status);    // Set angular arena
    _status = Walabot_SetThreshold(_threshold); _check_status(_status);                             // Set threshold

    if (_MTI == walabot::ACTIVATE_MTI)
    {
        _status = Walabot_SetDynamicImageFilter(FILTER_TYPE_MTI);                                   // Set motion target identification filter
        _check_status(_status);
    }
    _status = Walabot_Start(); _check_status(_status);                                              // Walabot start up
    _status = Walabot_StartCalibration(); _check_status(_status);                                   // Calibration
}

void walabot::disconnect()
/* Disconnect routine provides manual shut down means to modify the scanning profiles.*/
{
    Walabot_Stop(); Walabot_Disconnect();                                                           // Stop and disconnect
}

void walabot::set_phi(const ANGULAR_ARENA phi_min, const ANGULAR_ARENA phi_max, const ANGULAR_RESOL phi_res)
/* Configuration routines for walabot. Ensure disconnect is applied before adopting these coefficients
 * modification routines. They were for horizontal, perpendicular and depth parameters modification
 * separately.
 * Args:
 *      phi_min     : Minimum angular scan profile on phi
 *      phi_max     : Maximum angular scan profile on phi
 *      phi_res     : Angular resolution on phi*/
{
    WALABOT_RESULT _status;
    _phi_max = phi_max; _phi_min = phi_min; _phi_res = phi_res;
    _status = Walabot_SetArenaPhi(_phi_min, _phi_max, _phi_res); _check_status(_status);            // Set angular arena
}

void walabot::set_theta(const ANGULAR_ARENA theta_min, const ANGULAR_ARENA theta_max, const ANGULAR_RESOL theta_res)
/* Configuration routines for walabot. Ensure disconnect is applied before adopting these coefficients
 * modification routines. They were for horizontal, perpendicular and depth parameters modification
 * separately.
 * Args:
 *      theta_min   : Minimum angular scan profile on phi
 *      theta_max   : Maximum angular scan profile on phi
 *      theta_res   : Angular resolution on phi*/
{
    WALABOT_RESULT _status;
    _theta_min = theta_min; _theta_max = theta_max; _theta_res = theta_res;
    _status = Walabot_SetArenaTheta(_theta_min, _theta_max, _theta_res); _check_status(_status);    // Set angular arena
}

void walabot::set_r(const DEPTH_ARENA r_min, const DEPTH_ARENA r_max, const DEPTH_RESOL r_res)
/* Configuration routines for walabot. Ensure disconnect is applied before adopting these coefficients
 * modification routines. They were for horizontal, perpendicular and depth parameters modification
 * separately.
 * Args:
 *      theta_min   : Minimum angular scan profile on phi
 *      theta_max   : Maximum angular scan profile on phi
 *      theta_res   : Angular resolution on phi*/
{
    WALABOT_RESULT _status;
    _r_max = r_max; _r_min = r_min; _r_res = r_res;
    _status = Walabot_SetArenaR(_r_min, _r_max, _r_res); _check_status(_status);                    // Set scan depth
}

void walabot::set_thres(const THRES threshold)
/* Threshold reset routine. Threshold defines the minimum energy of detectable signals
 * Args:
 *      threshold   : Minimum energy of detectable signals*/
{
    WALABOT_RESULT _status;
    _threshold = threshold;
    _status = Walabot_SetThreshold(_threshold); _check_status(_status);                             // Set threshold
}

void walabot::set_filter(const FILTER filter)
/* Use of motion target identification filter.
 * Args:
 *      filter      : use MTI*/
{
    WALABOT_RESULT _status;
    _MTI = filter;
    _status = Walabot_SetDynamicImageFilter(FILTER_TYPE_MTI); _check_status(_status);               // Set motion target identification filter
}

void walabot::set_scan_profile(const APP_PROFILE _profile)
/* Scan profile configuration of walabot reset routine. Ensure disconnect is applied before call
 * this routine.
 * Args:
 *      _profile    : expected scanning profile*/
{
    WALABOT_RESULT _status;
    _status = Walabot_SetProfile(_profile); _check_status(_status);                                 // Scanning configuration
}

Mat & walabot::get_frame(const SCAN_PROF _scan_prof)
/* Getting specific single frame of projection of depth energy map. It can be set with scan profiles
 * , determinating projection to horizontal plane or perpendicular plane. The projection would be transformed
 * to opencv MAT, its dimensions are dependent on scanning configuration.
 * Args:
 *      _scan_prof  : scan profile
 * Returns:
 *      Energy projections in the form of opencv MAT*/
{
    WALABOT_RESULT _status;
    int _size_x; int _size_y; int _size_z;
    int * _canvas; double _energy;

    _status = Walabot_Trigger(); _check_status(_status);                                            // Start scanning
    _status = Walabot_GetRawImage(&_canvas, &_size_x, &_size_y, &_size_z, &_energy); _check_status(_status);
                                                                                                    // Getting raw data
    Mat _rawimg(_size_y*_size_z, _size_x, CV_32S, _canvas);
    _rawimg = _rawimg.inv();                                                                        // Store data with opencv MAT
    if (_scan_prof == walabot::SCAN_HORIZONTAL) _rawimg = _sum_horizontal(_rawimg, _size_x, _size_y, _size_z);
    else _rawimg = _sum_perpendicular(_rawimg, _size_x, _size_y, _size_z);                          // Projection
    _rawimg.release();
    return _rawimg;
}

void walabot::scan(const char * _save_dir, const int _frame)
/* Scan routine for walabot. It performs scanning operation of walabot and save the captured data to
 * save directory in the form of binary. There is no transformation in this routine, and no saving operation
 * during scanning to ensure highest rate. The number of saved files are determined by required frames. The
 * scanning process would terminate if it has reached maximum frames. The file name of saved files are named by
 * operation time.
 * Args:
 *      _save_dir  : saving directory
 *      _frame     : maximum frames*/
{
    WALABOT_RESULT _status;
    int _size_x; int _size_y; int _size_z; double _energy;
    const int _dir_len = strlen(_save_dir);
    int ** _canvas = new int*[_frame];
    clock_t _clock_list[_frame];
    auto _sig_file = new char[_dir_len + CLOCK_T_DECBITS + 1];                                      // Concatenate saving directory
    clock_t _time = clock();                                                                        // Counter

    for (int _count = 0; _count < _frame; ++_count)
    {
        _status = Walabot_Trigger(); _check_status(_status);                                        // Scanning and collect data
        _status = Walabot_GetRawImage(&_canvas[_count], &_size_x, &_size_y, &_size_z, &_energy); _check_status(_status);
        _time = clock(); _clock_list[_count] = _time;
    }

    int _sz[] = {_size_x, _size_y, _size_z};
    for (int _count = 0; _count < _frame; ++_count)                                                 // Save files
    {
        sprintf(_sig_file, "%s%10ld", _save_dir, _clock_list[_count]);
        _signal_write(_sig_file, _canvas[_count], _sz);
    }
}

void walabot::union_scan(const char * _save_dir, const int _frame)
/* Improved scan routine for walabot. It performs scanning operation of walabot and save the captured data to
 * save directory in the form of binary. There is no transformation in this routine, and no saving operation
 * during scanning to ensure highest rate. The number of saved files are determined by required frames. The
 * scanning process would terminate if it has reached maximum frames. The file name of saved files are named by
 * operation time. The scanning process proceed only when a person is detected, and a detector provides the signal
 * through a text file.
 * Args:
 *      _save_dir  : saving directory
 *      _frame     : maximum frames*/
{
    WALABOT_RESULT _status;
    int _size_x; int _size_y; int _size_z; double _energy;
    const int _dir_len = strlen(_save_dir);
    int ** _canvas = new int*[_frame];
    clock_t _clock_list[_frame];
    auto _sig_file = new char[_dir_len + CLOCK_T_DECBITS + 1];                                      // Concatenate saving directory
    auto _inter_file = new char[_dir_len + 11];                                                     // Interaction signal file
    sprintf(_inter_file, "%s%s", _save_dir, "inter.txt");
    clock_t _time = clock();                                                                        // Counter
    auto signal = new char[1]; signal[0] = '0';                                                     // Signal receiver

    while (1)
    {
        FILE * inter = fopen(_inter_file, "r");
        fread(signal, sizeof(char), 1, inter);
        fclose(inter);

        if (signal[0] == '0') continue;

        for (int _count = 0; _count < _frame; ++_count) {
            _status = Walabot_Trigger();
            _check_status(_status);                                        // Scanning and collect data
            _status = Walabot_GetRawImage(&_canvas[_count], &_size_x, &_size_y, &_size_z, &_energy);
            _check_status(_status);
            _time = clock();
            _clock_list[_count] = _time;
        }

        int _sz[] = {_size_x, _size_y, _size_z};
        for (int _count = 0; _count < _frame; ++_count)                                                 // Save files
        {
            sprintf(_sig_file, "%s%10ld", _save_dir, _clock_list[_count]);
            _signal_write(_sig_file, _canvas[_count], _sz);
        }

        signal[0] = 0;
        inter = fopen(_inter_file, "w");                                                                // Send signal
        fwrite("0", sizeof(char), 1, inter);
        fclose(inter);
    }
}

void walabot::_scan_test()
{
    WALABOT_RESULT _status;
    int _size_x; int _size_y; int _size_z;
    int * _canvas; double _energy;
    clock_t _start = clock(), _time = clock();

    while(1) {
        _status = Walabot_Trigger();
        _check_status(_status);
        _status = Walabot_GetRawImage(&_canvas, &_size_x, &_size_y, &_size_z, &_energy);
        _check_status(_status);
        Mat _rawimg(_size_y*_size_z, _size_x, CV_32S, _canvas);
        std::cout << _rawimg << std::endl;
        int _sz[] = {_size_x, _size_y, _size_z};
        std::cout << _energy << std::endl;
    }
}

void walabot::_check_status(WALABOT_RESULT & _status)
/* Check the running status of walabot. It throws runtime_error exception if any error occurs in the operation
 * process. Its parameter is got from routine process of walabot.
 * Args:
 *      _status      : running status*/
{
    if (_status == WALABOT_SUCCESS) return;

    const char * _error_str = Walabot_GetErrorString();
    throw *new std::runtime_error(_error_str);
}

int *** walabot::_get_canvas(const size_t & _x, const size_t & _y, const size_t & _z)
/* Allocate memory for tri-dimensional canvas to get raw images from walabot.
 * Args:
 *      _x          : Height dimension of the tensor
 *      _y          : Width dimension of the tensor
 *      _z          : Depth dimension of the tensor*/
{
    int *** _canvas = new int ** [_x];
    for (int _i = 0; _i < _x; ++_i)
    {
        _canvas[_i] = new int * [_y];
        for (int _j = 0; _j < _y; ++_j) _canvas[_i][_j] = new int[_z];
    }
    return _canvas;
}

void walabot::_signal_write(const char * _file, const int * _signal, const int * _sz)
/* Signal saving routine. The signal scalar would be saved in the form of binary files. The lenth are given
 * through calling. The file includes the dimensions of depth energy map and its corresponding energy of every
 * voxel. The dimensions are saved in the following form: (unsigned int)H W C, followed by energy scalar (int)
 * Args:
 *      _file       : file name
 *      _signal     : signal scalar
 *      _sz         : dimensions of map*/
{
    FILE * _sig_file = fopen(_file, "wb");
    const size_t _len = _sz[0]*_sz[1]*_sz[2];
    fwrite(_sz, sizeof(unsigned int), 3, _sig_file);
    fwrite(_signal, sizeof(int), _len, _sig_file);
    fclose(_sig_file);
}

Mat & walabot::_singal_read(const char * _sig_file)
/* Signal reading routine. The signal scalar would be saved in the form of binary files. The lenth are given
 * through calling. The file includes the dimensions of depth energy map and its corresponding energy of every
 * voxel. The dimensions are saved in the following form: (unsigned int)H W C, followed by energy scalar (int)
 * Args:
 *      _sig_file   : file name
*/
{
    FILE * _signal_ = fopen(_sig_file, "rb");
    auto _sz = new unsigned int[3];
    fread(_sz, sizeof(unsigned int), 3, _signal_);
    auto _signal = new int[_sz[0]*_sz[1]*_sz[2]];
    fread(_signal, sizeof(int), _sz[0]*_sz[1]*_sz[2], _signal_);
    Mat * _raw_img = new Mat(_sz[1]*_sz[2], _sz[0], CV_32S, _signal);
    fclose(_signal_);

    return *_raw_img;
}

void walabot::_delete_canvas(int *** _canvas, const size_t & _x, const size_t & _y)
/* Free memory for tri-dimensional canvas
 * Args:
 *      _x          : Height dimension of the tensor
 *      _y          : Width dimension of the tensor*/
{
    for (int _i = 0; _i < _x; ++_i)
    {
        for (int _j = 0; _j < _x; ++_j) delete[] _canvas[_i][_j];
        delete[] _canvas[_i];
    }
    delete[] _canvas;
}

Mat & walabot::_sum_horizontal(const Mat & _img, const size_t & _x, const size_t & _y, const size_t & _z)
{
    Mat * _sumimg = new Mat(Mat::zeros(_z, _y, CV_32S));

    for (int _k = 0; _k < _x; ++_k) {
        for (int _i = 0; _i < _z; ++_i)
            for (int _j = 0; _j < _y; ++_j) _sumimg->at<int>(_z - _i - 1, _j) += _img.at<int>(_k, _i * _y + _j);
    }

    return *_sumimg;
}

Mat & walabot::_sum_perpendicular(const Mat & _img, const size_t & _x, const size_t & _y, const size_t & _z)
{
    Mat * _sumimg = new Mat(Mat::zeros(_z, _x, CV_32S));

    for (int _k = 0; _k < _y; ++_k) {
        for (int _i = 0; _i < _z; ++_i)
            for (int _j = 0; _j < _x; ++_j) _sumimg->at<uchar>(_z - _i - 1, _j) += _img.at<uchar>(_j, _i *_y + _k);
    }

    return *_sumimg;
}