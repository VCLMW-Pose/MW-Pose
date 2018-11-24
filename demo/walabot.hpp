//     Created on Nov 24 15:59 2018
//
//     Author           : Shaoshu Yang
//     Email            : shaoshuyangseu@gmail.com
//     Last edit date   : Nov 24 24:00 2018
//
//South East University Automation College, 211189 Nanjing China

#ifndef DEMO_WALABOT_HPP
#define DEMO_WALABOT_HPP

#include <WalabotAPI.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <exception>
#include <stdexcept>

using namespace cv;

typedef double _angular_arena;                                                                      // Angular arena scale in angle
typedef double _depth_arena;                                                                        // Depth arena scale in centimeter
typedef double _depth_resol;                                                                        // Depth resolution in centimeter
typedef double _angular_resol;                                                                      // Angular resolution in angle
typedef bool _filter;
typedef int _scan_prof;                                                                             // Scan profile

class walabot
{
    const _scan_prof SCAN_HORIZONTAL = 0;                                                           // Project energies to horizontal plane
    const _scan_prof SCAN_PERPENDICULAR = 1;                                                        // Project energies to perpendicular plane

private:
    /* Basic coefficient of walabot scanning profile.*/
    _depth_arena _r_min;                                                                            // Minimum depth of scanning arena
    _depth_arena _r_max;                                                                            // Maximum depth of scanning arena
    _depth_resol _r_res;                                                                            // Depth spatial resolution

    _angular_arena _phi_min;                                                                        // Minimum phi angle
    _angular_arena _phi_max;                                                                        // Maximum phi angle
    _angular_resol _phi_res;                                                                        // Angular resolution of phi

    _angular_arena _theta_min;                                                                      // Minimum theta angle
    _angular_arena _theta_max;                                                                      // Maximum theta angle
    _angular_resol _theta_res;                                                                      // Angular resolution of theta

    _filter _MTI;                                                                                   // Applying dynamic reflector filter

public:
    /* Default constructor and destructor of walabot class. Walabot must be intialized with a
     * set of coefficients. The constructor routine would not initialize walabot, hence it can
     * not perform scanning immediately. The defualt destructor would disconncect walabot automatically.*/
    walabot(_depth_arena r_min, _depth_arena r_max, _depth_resol r_res, _angular_arena phi_min, _angular_arena phi_max,
                    _angular_resol phi_res, _angular_arena theta_min, _angular_arena theta_max, _angular_resol theta_res);
                                                                                                    // Defualt constructor
    ~walabot();                                                                                     // Defualt destructor

    /* State control routine for walabot. After applying start routine, walabot can perform scanning and
     * collecting images. disconnect routine provides manual shut down means to modify the scanning
     * profiles.*/
    bool start();                                                                                   // Start up
    bool disconnect();                                                                              // Shut down

    /* Configuration routines for walabot. Ensure disconnect is applied before adopting these coefficients
     * modification routines. They were for horizontal, perpendicular and depth parameters modification
     * separately. */
    bool set_phi(const _angular_arena phi_min, const _angular_arena phi_max, const _angular_resol phi_res);
    bool set_theta(const _angular_arena theta_min, const _angular_arena theta_max, const _angular_resol theta_res);
    bool set_r(const _depth_arena r_min, const _depth_arena r_max, const _depth_resol r_res);
                                                                                                    // Phi scale modification
                                                                                                    // Theta scale modification
                                                                                                    // Depth scale modification

    bool set_filter(const _filter filter);                                                          // Reset motion target identification filter

    /* Scanning routine for walabot. The direction of projection can be determined by the parameter scan_prof.
     * These routines stores collected data in cv::Mat. */
    Mat & get_frame(const _scan_prof scan_prof);                                                    // Get single frame
    Mat * scan(const _scan_prof scan_prof);                                                         // Get projections over both planes

    /* Private test routines. Provide visual outputs of scanning profiles*/
    void _scan_test(const _scan_prof scan_prof);                                                    // Scan test
};

#endif //DEMO_WALABOT_HPP
