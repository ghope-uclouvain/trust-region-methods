#ifndef OPT_TR_H
#define OPT_TR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct opt_tr opt_tr;



// ░█▀▀░█░█░█▀█░█▀▀░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█▀▀░█░█░█░█░█░░░░█░░░█░░█░█░█░█░▀▀█
// ░▀░░░▀▀▀░▀░▀░▀▀▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

// returns f(x)
typedef void (*opt_tr_fun_t)( 
    const double *x, double *fx, void *userdata);

// returns the gradient at x
typedef void (*opt_tr_grad_t)(
    const double *x, double *g,  void *userdata);

// returns the hessian at x
typedef void (*opt_tr_hess_vec_t)(
    const double *x, const double *v, double *Hv, void *userdata);



// ░█░█░▀█▀░█▀▄░▀█▀░█░█░█▀█░█░░░░░█▄█░█▀▀░▀█▀░█░█░█▀█░█▀▄░█▀▀
// ░▀▄▀░░█░░█▀▄░░█░░█░█░█▀█░█░░░░░█░█░█▀▀░░█░░█▀█░█░█░█░█░▀▀█
// ░░▀░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░░░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀░░▀▀▀
                                                                           
// determines the next step
typedef int  (*opt_tr_step_t)(opt_tr *tr, double* expected_improvement);

// calculates trust region radius for next step
typedef int  (*opt_tr_update_radius_t)(opt_tr *tr, double ratio);

// decides what do with next step
typedef int  (*opt_tr_accept_step_t)(opt_tr *tr, double ratio);

// actually updates next step
typedef int  (*opt_tr_qn_update_t)(opt_tr *tr);



// ░█▀▀░█▀█░█▀█░█▀▀░▀█▀░█▀▀░█░█░█▀▄░█▀█░▀█▀░▀█▀░█▀█░█▀█░█▀▀
// ░█░░░█░█░█░█░█▀▀░░█░░█░█░█░█░█▀▄░█▀█░░█░░░█░░█░█░█░█░▀▀█
// ░▀▀▀░▀▀▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░▀░▀░▀░░▀░░▀▀▀░▀▀▀░▀░▀░▀▀▀

typedef struct {
    size_t max_iters;        // maximum iterations
    int max_time;            // maximum amount of computation time (seconds)
    double starting_radius;  // starting trust region radius
    double radius_max;       // maximum trust region radius
    double gtol;             // gradient tolerance
    double xtol;             // step size tolerance
    bool verbose;            // whether user wants status updates or not
} opt_tr_options_t;



// ░█▀▄░█▀▀░█▀▀░█░█░█░░░▀█▀░█▀▀
// ░█▀▄░█▀▀░▀▀█░█░█░█░░░░█░░▀▀█
// ░▀░▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀░░▀░░▀▀▀
typedef struct {
    size_t iterations;       // number of iterations to reach solution
    double final_radius;     // radius of final trust region
    double x;                // value of x at the solution
    double fval;             // value of the function at the solution
    double time;             // time to reach solution (seconds)
    int status;              // error status code
} opt_tr_result_t;



// ░█░█░▀█▀░█▀▄░▀█▀░█░█░█▀█░█░░░░░▀█▀░█▀█░█▀▄░█░░░█▀▀
// ░▀▄▀░░█░░█▀▄░░█░░█░█░█▀█░█░░░░░░█░░█▀█░█▀▄░█░░░█▀▀
// ░░▀░░▀▀▀░▀░▀░░▀░░▀▀▀░▀░▀░▀▀▀░░░░▀░░▀░▀░▀▀░░▀▀▀░▀▀▀

typedef struct {
    opt_tr_step_t          step;
    opt_tr_update_radius_t update_radius;
    opt_tr_accept_step_t   accept_step;
    opt_tr_qn_update_t     qn_update;      // quasi-newton update
} opt_tr_vtable_t;



//    ░█▀▄░█▀█░█▀▀░█▀▀░░░█▀▀░█░░░█▀█░█▀▀░█▀▀
//    ░█▀▄░█▀█░▀▀█░█▀▀░░░█░░░█░░░█▀█░▀▀█░▀▀█
//    ░▀▀░░▀░▀░▀▀▀░▀▀▀░░░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀
struct opt_tr {

    
    size_t n; // problem size

    void *userdata; // user data
    opt_tr_fun_t  fun; // function
    opt_tr_grad_t grad; // gradient
    opt_tr_hess_vec_t hess_vec; //hessian

    
    double* x;
    double* test_x; 
    double* g;
    double* work;
    double radius;

    
    opt_tr_options_t config;

    
    const opt_tr_vtable_t *vptr;
};


// constructor
int opt_tr_init(
    opt_tr *tr,
    size_t n,
    opt_tr_fun_t fun,
    opt_tr_grad_t grad,
    opt_tr_hess_vec_t hess_vec,
    void *userdata
);

// destructor
void opt_tr_base_free(opt_tr *tr);

// options setter
void opt_tr_set_options(opt_tr *tr, const opt_tr_options_t *opts);

// solver
int opt_tr_solve(opt_tr *tr, double *x0, opt_tr_result_t *res);

#ifdef __cplusplus
}
#endif

#endif 
