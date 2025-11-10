#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "../include/opt_tr.h"


static double current_time(void)
{
    struct timespec ts;
#if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    ts.tv_sec  = time(NULL);
    ts.tv_nsec = 0;
#endif
    return (double)ts.tv_sec + 1e-9 * ts.tv_nsec;
}


int opt_tr_init(
    opt_tr *tr,
    size_t n,
    opt_tr_fun_t fun,
    opt_tr_grad_t grad,
    opt_tr_hess_vec_t hess_vec,
    void *userdata
)
{
    if (!tr || !fun || !grad) 
        return -1; // invalid input

    tr->n         = n;
    tr->fun       = fun;
    tr->vptr      = NULL;      // to be defined by the solver

    // default values
    tr->config.max_iters       = 100;
    tr->config.max_time        = UINT32_MAX; // if the problem can't be solved in 136 years, you might have other issues
    tr->config.starting_radius = 1.0;
    tr->config.radius_max      = UINT32_MAX; 
    tr->config.gtol            = 1e-10;
    tr->config.xtol            = 1e-10;
    tr->config.verbose         = 0;
    tr->radius = tr->config.starting_radius;

    /* allocate working memory */
    tr->x_k    = (double*)calloc(n, sizeof(double));
    tr->test_x    = (double*)calloc(n, sizeof(double));
    tr->g    = (double*)calloc(n, sizeof(double));
    tr->work = (double*)calloc(n, sizeof(double));

    if (!tr->x_k || !tr->g || !tr->work) {
        opt_tr_free(tr);
        return -1;
    }

    return 0;
}


// destructor
void opt_tr_free(opt_tr *tr)
{
    if (!tr) return;
    free(tr->x_k);
    free(tr->g);
    free(tr->work);

    tr->x_k    = NULL;
    tr->g    = NULL;
    tr->work = NULL;
    tr->vptr = NULL;
}


// options setter
void opt_tr_set_options(opt_tr *tr, const opt_tr_options_t *config)
{
    if (!tr || !config) return;
    tr->config = *config;

    /* update radius to starting value */
    tr->radius = tr->config.starting_radius;
}


// actual trust region solver
int opt_tr_solve(opt_tr *tr, double *x0, opt_tr_result_t *res)
{
    if(tr->config.verbose){
        print("Beginning solve...\n");
    }
    if (!tr || !x0 || !res || !tr->vptr) {
        return -1;   // missing vtable or null inputs
    }

    const size_t n = tr->n;


    // copy x0 -> internal state 
    memcpy(tr->x_k, x0, n * sizeof(long double));

    // compute f(x0) + gradient 
    double fx;
    double test_fx;
    tr->fun(tr->x_k, &fx);
    // tr->grad(tr->x, tr->g);

    double gnorm = 0.0;

    size_t iter = 0;
    double t0   = now_seconds();

    for (iter = 0; iter < tr->config.max_iters; ++iter) {
        if(tr->config.verbose){
            print("\nSTART OF ITERATION %d:", iter + 1);
        }
        double elapsed = now_seconds() - t0;
        if (elapsed > tr->config.max_time) {
            if(tr->config.verbose){
                print("   Solve exceeded maximum time! Ending.");
            }
            break;
        }
        
        if(tr->config.verbose){
            print("   Current runtime: %d\n", elapsed);
            print("   Starting convergence test...\n");
        }

        // convergence test
        gnorm = 0.0;
        for (size_t i = 0; i < n; ++i)
            gnorm += tr->g[i] * tr->g[i];
        gnorm = sqrt(gnorm);

        if (gnorm <= tr->config.gtol){
            if(tr->config.verbose){
                print("   Solution within gradience tolerance! Ending.");
            } 
            break;
        }

        if(tr->config.verbose){
            print("   Computing trial step...");
        } 

        
        // computing trial step
        double* expected_improvement;
        if (tr->vptr->step) {
            int rc = tr->vptr->step(tr, expected_improvement);
            if (rc != 0) break;
        } else {
            // virtual function is missing
            break;
        }

        tr->fun(tr->test_x, &test_fx);
        
        double ratio = (fx - test_fx) / *expected_improvement;

        // update trust region radius
        if (tr->vptr->update_radius) {
            int rc = tr->vptr->update_radius(tr, ratio);
            if (rc != 0) break;
        }

        // choose to accept or reject step
        if (tr->vptr->accept_step) {
            int rc = tr->vptr->accept_step(tr, ratio);
            if (rc != 0) break;
        }

        // update quasi-newton approximation
        if (tr->vptr->qn_update) {
            int rc = tr->vptr->qn_update(tr);
            if (rc != 0) break;
        }

        // recompute f and g
        tr->fun(tr->x_k, &fx);
        // tr->grad(tr->x_k, tr->g);


        if (tr->config.xtol > 0.0) {
            break;
        }
    }

    // write results
    res->iterations   = iter;
    res->final_radius = tr->radius;
    res->x = *tr->x_k;
    res->fval = fx;
    res->time = now_seconds() - t0;
    res->status = 0;

    // write final x back into x0
    memcpy(x0, tr->x_k, n * sizeof(double));

    return 0;
}
