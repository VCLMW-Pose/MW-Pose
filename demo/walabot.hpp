//     Created on Nov 24 15:59 2018
//
//     Author           : Shaoshu Yang
//     Email            : shaoshuyangseu@gmail.com
//     Last edit date   : Nov 25 9:31 2018
//
//South East University Automation College, 211189 Nanjing China

#ifndef DEMO_WALABOT_HPP
#define DEMO_WALABOT_HPP
#ifdef __LINUX__
#define CONFIG_FILE_PATH "/etc/walabotsdk.conf"
#else
#define CONFIG_FILE_PATH "C:\\Program Files\\Walabot\\WalabotSDK\\bin\\.config"
#endif

#include "WalabotAPI.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <exception>
#include <stdexcept>

using namespace cv;

typedef double ANGULAR_ARENA;                                                                      // Angular arena scale in angle
typedef double DEPTH_ARENA;                                                                        // Depth arena scale in centimeter
typedef double DEPTH_RESOL;                                                                        // Depth resolution in centimeter
typedef double ANGULAR_RESOL;                                                                      // Angular resolution in angle
typedef bool FILTER;
typedef int SCAN_PROF;                                                                             // Scan profile
typedef double THRES;                                                                              // Threshold

class walabot
{
    const SCAN_PROF SCAN_HORIZONTAL = 0;                                                           // Project energies to horizontal plane
    const SCAN_PROF SCAN_PERPENDICULAR = 1;                                                        // Project energies to perpendicular plane
    const FILTER NO_APPLYING_MTI = false;                                                          // MTI filter constant
    const FILTER ACTIVATE_MTI = true;                                                              // MTI filter constant
    const int CLOCK_T_DECBITS = 100;

private:
    /* Basic coefficient of walabot scanning profile.*/
    DEPTH_ARENA _r_min;                                                                            // Minimum depth of scanning arena
    DEPTH_ARENA _r_max;                                                                            // Maximum depth of scanning arena
    DEPTH_RESOL _r_res;                                                                            // Depth spatial resolution

    ANGULAR_ARENA _phi_min;                                                                        // Minimum phi angle
    ANGULAR_ARENA _phi_max;                                                                        // Maximum phi angle
    ANGULAR_RESOL _phi_res;                                                                        // Angular resolution of phi

    ANGULAR_ARENA _theta_min;                                                                      // Minimum theta angle
    ANGULAR_ARENA _theta_max;                                                                      // Maximum theta angle
    ANGULAR_RESOL _theta_res;                                                                      // Angular resolution of theta

    FILTER _MTI;                                                                                   // Applying dynamic reflector filter
    THRES _threshold;                                                                              // Threshold of detectable signal

public:
    /* Default constructor and destructor of walabot class. Walabot must be intialized with a
     * set of coefficients. The constructor routine would not initialize walabot, hence it can
     * not perform scanning immediately. The defualt destructor would disconncect walabot automatically.*/
    walabot(DEPTH_ARENA r_min, DEPTH_ARENA r_max, DEPTH_RESOL r_res, ANGULAR_ARENA phi_min, ANGULAR_ARENA phi_max,
            ANGULAR_RESOL phi_res, ANGULAR_ARENA theta_min, ANGULAR_ARENA theta_max, ANGULAR_RESOL theta_res, FILTER filter,
            THRES threshold);                                                                       // Defualt constructor
    ~walabot();                                                                                     // Defualt destructor

    /* State control routine for walabot. After applying start routine, walabot can perform scanning and
     * collecting images. disconnect routine provides manual shut down means to modify the scanning
     * profiles.*/
    void start();                                                                                   // Start up
    void disconnect();                                                                              // Shut down

    /* Configuration routines for walabot. Ensure disconnect is applied before adopting these coefficients
     * modification routines. They were for horizontal, perpendicular and depth parameters modification
     * separately. */
    void set_phi(ANGULAR_ARENA phi_min, ANGULAR_ARENA phi_max, ANGULAR_RESOL phi_res);
    void set_theta(ANGULAR_ARENA theta_min, ANGULAR_ARENA theta_max, ANGULAR_RESOL theta_res);
    void set_r(DEPTH_ARENA r_min, DEPTH_ARENA r_max, DEPTH_RESOL r_res);
                                                                                                    // Phi scale modification
                                                                                                    // Theta scale modification
                                                                                                    // Depth scale modification
    void set_thres(THRES threshold);                                                                // Threshold modification
    void set_filter(FILTER filter);                                                                 // Reset motion target identification filter
    void set_scan_profile(APP_PROFILE profile);                                                     // Reset scanning profile

    /* Scanning routine for walabot. The direction of projection can be determined by the parameter scan_prof.
     * These routines stores collected data in cv::Mat. */
    Mat & get_frame(SCAN_PROF scan_prof);                                                           // Get single frame
    void scan(const char * _save_dir, const int _frames);                                           // Get projections over both planes
    void union_scan(const char * _save_dir, const int _frame);                                      // Scan by interacting with python human detector

    /* Private test routines. Provide visual outputs of scanning profiles*/
    void _scan_test();                                                                              // Scan test
    void _check_status(WALABOT_RESULT & _status);                                                   // Check running status of walabot
    int *** _get_canvas(const size_t & _x, const size_t & _y, const size_t & _z);                   // Get blank canvas
    void _delete_canvas(int *** _canvas, const size_t & _x, const size_t & _y);                     // Free memory of canvas
    Mat & _sum_horizontal(const Mat & _img, const size_t & _x, const size_t & _y, const size_t & _z);
                                                                                                    // Projection to horizontal plane
    Mat & _sum_perpendicular(const Mat & _img, const size_t & _x, const size_t & _y, const size_t & _z);
                                                                                                    // Projection to perpendicular plane

    /* Raw signal save and read routine*/
    void _signal_write(const char * _file, const int * _signal, const int * _sz);                   // Signal write
    Mat & _singal_read(const char * _sig_file);                                                     // Signal read
};

#endif //DEMO_WALABOT_HPP
