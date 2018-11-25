//     Created on Nov 24 23:45 2018
//
//     Author           : Shaoshu Yang
//     Email            : shaoshuyangseu@gmail.com
//     Last edit date   : Nov 24 24:00 2018
//
//South East University Automation College, 211189 Nanjing China

#include "walabot.hpp"

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
    _status = Walabot_Initialize(CONFIG_FILE_PATH);                                                 // Initialize
    _check_status(_status);

    _status = Walabot_ConnectAny();                                                                 // Connect to walabot
    _check_status(_status);

    _status = Walabot_SetProfile(PROF_SENSOR);                                                      // Scanning configuration
    _check_status(_status);

    _status = Walabot_SetArenaR(_r_min, _r_max, _r_res);                                            // Set scan depth
    _check_status(_status);

    _status = Walabot_SetArenaPhi(_phi_min, _phi_max, _phi_res);                                    // Set angular arena
    _check_status(_status);

    _status = Walabot_SetArenaTheta(_theta_min, _theta_max, _theta_res);                            // Set angular arena
    _check_status(_status);

    _status = Walabot_SetThreshold(_threshold);                                                     // Set threshold
    _check_status(_status);

    if (_MTI == walabot::ACTIVATE_MTI)
    {
        _status = Walabot_SetDynamicImageFilter(FILTER_TYPE_MTI);                                   // Set motion target identification filter
        _check_status(_status);
    }

    _status = Walabot_Start();                                                                      // Walabot start up
    _check_status(_status);

    _status = Walabot_StartCalibration();                                                           // Calibration
    _check_status(_status);
}

void walabot::disconnect()
/* Disconnect routine provides manual shut down means to modify the scanning profiles.*/
{
    Walabot_Stop();
    Walabot_Disconnect();
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
    _status = Walabot_SetArenaPhi(_phi_min, _phi_max, _phi_res);                                    // Set angular arena
    _check_status(_status);
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
    _status = Walabot_SetArenaTheta(_theta_min, _theta_max, _theta_res);                            // Set angular arena
    _check_status(_status);

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
    _status = Walabot_SetArenaR(_r_min, _r_max, _r_res);                                            // Set scan depth
    _check_status(_status);
}

void walabot::set_thres(const THRES threshold)
/* Threshold reset routine. Threshold defines the minimum energy of detectable signals
 * Args:
 *      threshold   : Minimum energy of detectable signals*/
{
    WALABOT_RESULT _status;
    _threshold = threshold;
    _status = Walabot_SetThreshold(_threshold);                                                     // Set threshold
    _check_status(_status);
}

void walabot::set_filter(const FILTER filter)
/* Use of motion target identification filter.
 * Args:
 *      filter      : use MTI*/
{
    WALABOT_RESULT _status;
    _MTI = filter;
    _status = Walabot_SetDynamicImageFilter(FILTER_TYPE_MTI);                                       // Set motion target identification filter
    _check_status(_status);
}

void walabot::set_scan_profile(const APP_PROFILE _profile)
/* Scan profile configuration of walabot reset routine. Ensure disconnect is applied before call
 * this routine.
 * Args:
 *      _profile    : expected scanning profile*/
{
    WALABOT_RESULT _status;
    _status = Walabot_SetProfile(_profile);                                                         // Scanning configuration
    _check_status(_status);
}

Mat & walabot::get_frame(const SCAN_PROF _scan_prof)
{
    WALABOT_RESULT _status;
    int _size_x;
    int _size_y;
    int _size_z;
    int * _canvas;
    double _energy;

    _status = Walabot_Trigger();
    _check_status(_status);
    _status = Walabot_GetRawImage(&_canvas, &_size_x, &_size_y, &_size_z, &_energy);
    int _size[] = {_size_x, _size_y, _size_z};
    _check_status(_status);

    Mat _rawimg(3, _size, CV_8U, _canvas);

}

/*Mat * walabot::scan(const SCAN_PROF scan_prof)
{

}

void walabot::_scan_test(const SCAN_PROF scan_prof)
{

}*/

void walabot::_check_status(WALABOT_RESULT & _status)
/* Check the running status of walabot. It throws runtime_error exception if any error occurs in the operation
 * process. Its parameter is got from routine process of walabot.
 * Args:
 *      _status      : running status*/
{
    if (_status == WALABOT_SUCCESS)
        return;

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
        for (int _j = 0; _j < _y; ++_j)
            _canvas[_i][_j] = new int[_z];
    }
    return _canvas;
}

void walabot::_delete_canvas(int *** _canvas, const size_t & _x, const size_t & _y)
/* Free memory for tri-dimensional canvas
 * Args:
 *      _x          : Height dimension of the tensor
 *      _y          : Width dimension of the tensor*/
{
    for (int _i = 0; _i < _x; ++_i)
    {
        for (int _j = 0; _j < _x; ++_j)
            delete[] _canvas[_i][_j];
        delete[] _canvas[_i];
    }
    delete[] _canvas;
}

/*Mat & _sum_horizontal(const Mat & _img, const size_t & _x, const size_t & _y)
{
}

Mat & _sum_perpendicular(const Mat & _img, const size_t & _x, const size_t & _y)
{

}*/