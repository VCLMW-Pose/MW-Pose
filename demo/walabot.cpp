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

bool walabot::start()
/* State control routine for walabot. After applying start routine, walabot can perform scanning and
     * collecting images.*/
{
    WALABOT_RESULT _status;
    _status = Walabot_Initialize(CONFIG_FILE_PATH);
    _check_status(_status);

    _status = Walabot_ConnectAny();
    _check_status(_status);

    _status = Walabot_SetProfile(PROF_SENSOR);
    _check_status(_status);

    _status = Walabot_SetArenaR(_r_min, _r_max, _r_res);
    _check_status(_status);

    _status = Walabot_SetArenaPhi(_phi_min, _phi_max, _phi_res);
    _check_status(_status);

    _status = Walabot_SetArenaTheta(_theta_min, _theta_max, _theta_res);
    _check_status(_status);

    if (_MTI == true)
    {
        _status = Walabot_SetDynamicImageFilter(FILTER_TYPE_MTI);
        _check_status(_status);
    }

    Walabot_Start();
    Walabot_StartCalibration();
}

bool walabot::disconnect()
/* Disconnect routine provides manual shut down means to modify the scanning profiles.*/
{
    Walabot_Stop();
    Walabot_Disconnect();
}

bool walabot::set_phi(const ANGULAR_ARENA phi_min, const ANGULAR_ARENA phi_max, const ANGULAR_RESOL phi_res)
/* Configuration routines for walabot. Ensure disconnect is applied before adopting these coefficients
 * modification routines. They were for horizontal, perpendicular and depth parameters modification
 * separately.
 * Args:
 *      phi_min     : Minimum angular scan profile on phi
 *      phi_max     : Maximum angular scan profile on phi
 *      phi_res     : Angular resolution on phi*/
{
    _phi_max = phi_max; _phi_min = phi_min; _phi_res = phi_res;
}

bool walabot::set_theta(const ANGULAR_ARENA theta_min, const ANGULAR_ARENA theta_max, const ANGULAR_RESOL theta_res)
/* Configuration routines for walabot. Ensure disconnect is applied before adopting these coefficients
 * modification routines. They were for horizontal, perpendicular and depth parameters modification
 * separately.
 * Args:
 *      theta_min   : Minimum angular scan profile on phi
 *      theta_max   : Maximum angular scan profile on phi
 *      theta_res   : Angular resolution on phi*/
{
    _theta_min = theta_min; _theta_max = theta_max; _theta_res = theta_res;
}

bool walabot::set_r(const DEPTH_ARENA r_min, const DEPTH_ARENA r_max, const DEPTH_RESOL r_res)
/* Configuration routines for walabot. Ensure disconnect is applied before adopting these coefficients
 * modification routines. They were for horizontal, perpendicular and depth parameters modification
 * separately.
 * Args:
 *      theta_min   : Minimum angular scan profile on phi
 *      theta_max   : Maximum angular scan profile on phi
 *      theta_res   : Angular resolution on phi*/
{
    _r_max = r_max; _r_min = r_min; _r_res = r_res;
}

bool walabot::set_thres(const THRES threshold)
/* Threshold reset routine. Threshold defines the minimum energy of detectable signals
 * Args:
 *      threshold   : Minimum energy of detectable signals*/
{
    _threshold = threshold;
}

bool walabot::set_filter(const FILTER filter)
/* Use of motion target identification filter.
 * Args:
 *      filter      : use MTI*/
{
    _MTI = filter;
}

Mat & walabot::get_frame(const SCAN_PROF scan_prof)
{

}

Mat * walabot::scan(const SCAN_PROF scan_prof)
{

}

void walabot::_scan_test(const SCAN_PROF scan_prof)
{

}

void walabot::_check_status(WALABOT_RESULT & _status)
{
    if (_status == WALABOT_SUCCESS)
        return;

    const char * _error_str = Walabot_GetErrorString();
    throw *new std::runtime_error(_error_str);
}