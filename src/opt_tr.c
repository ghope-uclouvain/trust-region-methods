#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

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
    size_t m,
    size_t n,
    double* x_k,
    double* test_k,
    double* gradient,
    double radius,
    double* c_k,
    opt_tr_options_t config,
    opt_tr_vtable_t *vptr
)
{
    tr->n = n;
    tr->m = m;

    tr->config.max_iters       = 100;
    tr->config.max_time        = (double)UINT32_MAX;
    tr->config.starting_radius = 1.0;
    tr->config.radius_max      = (double)UINT32_MAX;
    tr->config.gtol            = 1e-10;
    tr->config.xtol            = 1e-10;
    tr->config.verbose         = 0;

    tr->x_k     = x_k;
    tr->test_x  = test_k;
    tr->gradient = gradient;
    tr->radius   = radius;
    tr->c_k      = c_k;

    tr->config = config;
    tr->vptr   = vptr;

    return 0;
}


void opt_tr_free(opt_tr *tr)
{
    (void)tr;
}


/* options setter */
void opt_tr_set_options(opt_tr *tr, const opt_tr_options_t *config)
{
    if (!tr || !config) return;
    tr->config = *config;
    tr->radius = tr->config.starting_radius;
}


int opt_tr_solve(opt_tr *tr, double *x0, opt_tr_result_t *res)
{
    if (!tr || !x0 || !res || !tr->vptr)
        return -1;

    if (tr->config.verbose) {
        printf("Beginning solve...\n");
    }

    const size_t n = tr->n;

    memcpy(tr->x_k, x0, n * sizeof(double));

    double fx;
    tr->vptr->fun(tr->x_k, &fx);

    double gnorm = 0.0;
    size_t iter = 0;
    double t0   = current_time();

    for (iter = 0; iter < tr->config.max_iters; ++iter) {

        if (tr->config.verbose) {
            printf("\nSTART OF ITERATION %zu:\n", iter + 1);
        }

        double elapsed = current_time() - t0;
        if (elapsed > tr->config.max_time) {
            if (tr->config.verbose) {
                printf("   Solve exceeded maximum time! Ending.\n");
            }
            break;
        }

        if (tr->config.verbose) {
            printf("   Current runtime: %f\n", elapsed);
            printf("   Starting convergence test...\n");
        }

        gnorm = 0.0;
        for (size_t i = 0; i < n; ++i)
            gnorm += tr->gradient[i] * tr->gradient[i];

        gnorm = sqrt(gnorm);

        if (gnorm <= tr->config.gtol) {
            if (tr->config.verbose) {
                printf("   Solution within gradient tolerance! Ending.\n");
            }
            break;
        }

        if (tr->config.verbose) {
            printf("   Computing trial step...\n");
        }

        double expected_improvement = 1.0;
        if (tr->vptr->step) {
            int rc = tr->vptr->step(tr, &expected_improvement);
            if (rc != 0) break;
        } else {
            break;
        }

        double test_fx;
        tr->vptr->fun(tr->test_x, &test_fx);

        double ratio = (fx - test_fx) / expected_improvement;

        if (tr->vptr->update_radius) {
            int rc = tr->vptr->update_radius(tr, ratio);
            if (rc != 0) break;
        }

        if (tr->vptr->accept_step) {
            int rc = tr->vptr->accept_step(tr, ratio);
            if (rc != 0) break;
        }

        if (tr->vptr->qn_update) {
            int rc = tr->vptr->qn_update(tr);
            if (rc != 0) break;
        }

        tr->vptr->fun(tr->x_k, &fx);

        if (tr->config.xtol > 0.0) {
            break;
        }
    }

    res->iterations   = iter;
    res->final_radius = tr->radius;
    res->fval         = fx;
    res->time         = current_time() - t0;
    res->status       = 0;

    memcpy(x0, tr->x_k, n * sizeof(double));

    return 0;
}
