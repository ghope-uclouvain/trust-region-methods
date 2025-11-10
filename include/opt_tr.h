#ifndef OPT_TR_H
#define OPT_TR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

typedef struct opt_tr opt_tr;

// returns f(x)
typedef void (*opt_tr_fun_t)( 
    const long double* x, long double* fx);

// returns the gradient at x
typedef void (*opt_tr_grad_t)(
    const long double* x, long double* g);

// returns the hessian at x
typedef void (*opt_tr_hess_vec_t)(
    const long double* x, const long double* v, long double** Hv);

// returns the constraints function at x
typedef void (*opt_tr_constraints_t)(
    const long double* x, const long double* cx);

  
// determines the next step
typedef int  (*opt_tr_step_t)(opt_tr* tr, long double* expected_improvement);

// calculates trust region radius for next step
typedef int  (*opt_tr_update_radius_t)(opt_tr* tr, long double ratio);

// decides what do with next step
typedef int  (*opt_tr_accept_step_t)(opt_tr* tr, long double ratio);

// actually updates next step
typedef int  (*opt_tr_qn_update_t)(opt_tr* tr);

// returns the constraints function at x
typedef int (*opt_tr_constraints_t)(long double* cx);


typedef struct {
    size_t max_iters;        // maximum iterations
    int max_time;            // maximum amount of computation time (seconds)
    long double starting_radius;  // starting trust region radius
    long double radius_max;       // maximum trust region radius
    long double gtol;             // gradient tolerance
    long double xtol;             // step size tolerance
    bool verbose;            // whether user wants status updates or not
} opt_tr_options_t;



typedef struct {
    size_t iterations;       // number of iterations to reach solution
    long double final_radius;     // radius of final trust region
    long double x;                // value of x at the solution
    long double fval;             // value of the function at the solution
    long double time;             // time to reach solution (seconds)
    int status;              // error status code
} opt_tr_result_t;


typedef struct {
    opt_tr_step_t          step;
    opt_tr_update_radius_t update_radius;
    opt_tr_accept_step_t   accept_step;
    opt_tr_grad_t          calc_grad;      // method for calculating gradient
    opt_tr_qn_update_t     qn_update;      // quasi-newton update
} opt_tr_vtable_t;


struct opt_tr {
    opt_tr_fun_t  fun; // function

    size_t n; // dimensionality of problem

    opt_tr_constraints_t cons; // constraints of the problem

    opt_tr_constraints_t constraints; //constraints

    
    long double* x_k;
    long double* test_x; 
    long double* g;
    long double* work;
    long double radius;
    
    opt_tr_options_t config;

    const opt_tr_vtable_t *vptr;
};

// constructor
int opt_tr_init(
    opt_tr *tr,
    size_t n
);

// destructor
void opt_tr_base_free(opt_tr *tr);

// options setter
void opt_tr_set_options(opt_tr *tr, const opt_tr_options_t *opts);

// solver
int opt_tr_solve(opt_tr *tr, long double *x0, opt_tr_result_t *res);

#ifdef __cplusplus
}
#endif

#endif 
// constructor
