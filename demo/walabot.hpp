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

typedef double _angular_arena;
typedef double _depth_arena;
typedef double _depth_resol;
typedef double _angular_resol;
typedef bool _filter;

class walabot
{
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
    walabot(_depth_arena r_min, _depth_arena r_max, _depth_resol r_res, _angular_arena phi_min, _angular_arena phi_max,
                    _angular_resol phi_res, _angular_arena theta_min, _angular_arena theta_max, _angular_resol theta_res);
    ~walabot();
    
    bool start();
    bool disconnect();
    
    bool set_phi();
    bool set_theta();
    bool set_r();

    bool

};


#endif //DEMO_WALABOT_HPP
