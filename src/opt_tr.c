#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdio.h>

#include "../include/opt_tr.h"

/* -----------------------------------------------------------
   IMPORTANT NOTE:

   This function does ONLY the tasks common to all* trust region methods

   - copy x0 to internal x_k
   - evaluate f(x_k), g(x_k), c(x_k) once at start
   - main loop:
       * check time limit
       * compute ||g_k||
       * stop if ||g_k|| <= gtol
       * call the method-specific step(tr, &predicted)

   The method-specific step is
   responsible for:

   - computing a trial step s_k
   - computing and using a merit function ψ
   - computing the ratio ρ
   - calling update_radius(tr, ρ)
   - calling accept_step(tr, ρ)
   - calling qn_update(tr)

   * I am sure there are some methods that would require you to modify this file
   ----------------------------------------------------------- */

// for timing things
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

// constructor
int opt_tr_init(
    opt_tr *tr,
    size_t m,
    size_t n,
    double *x_k,
    double *test_k,
    double *gradient,
    double radius,
    double *c_k,
    opt_tr_options_t config,
    opt_tr_vtable_t *vptr
)
{
    if (!tr || !x_k || !test_k || !gradient || !vptr)
        return -1;

    tr->n = n;
    tr->m = m;

    /* Default options; caller may override via config. */
    tr->config.max_iters       = 100;
    tr->config.max_time        = (double)UINT32_MAX;
    tr->config.starting_radius = 1.0;
    tr->config.radius_max      = (double)UINT32_MAX;
    tr->config.gtol            = 1e-10;
    tr->config.xtol            = 0.0;          /* not enforced here */
    tr->config.verbose         = 0;

    tr->x_k      = x_k;
    tr->test_x   = test_k;
    tr->gradient = gradient;
    tr->radius   = radius;
    tr->c_k      = c_k;

    tr->config   = config;
    tr->vptr     = vptr;

    return 0;
}

void opt_tr_free(opt_tr *tr)
{
    (void)tr;
}

// set options
void opt_tr_set_options(opt_tr *tr, const opt_tr_options_t *config)
{
    if (!tr || !config) return;
    tr->config = *config;
    tr->radius = tr->config.starting_radius;
}


int opt_tr_solve(opt_tr *tr, double *x0, opt_tr_result_t *res)
{
    if (!tr || !x0 || !res || !tr->vptr || !tr->vptr->fun)
        return -1;

    const size_t n = tr->n;

    if (tr->config.verbose) {
        printf("Beginning trust-region solve (generic core)...\n");
    }

    // initialize x_k from user-defined starting point */
    memcpy(tr->x_k, x0, n * sizeof(double));

    // evaluate f, g, c at x_k once at the start
    double fx = 0.0;
    tr->vptr->fun(tr->x_k, &fx);

    if (tr->vptr->grad)
        tr->vptr->grad(tr->x_k, tr->gradient);

    if (tr->m > 0 && tr->vptr->constraints)
        tr->vptr->constraints(tr->x_k, tr->c_k);

    size_t iter = 0;
    double t0   = current_time();

    for (iter = 0; iter < tr->config.max_iters; ++iter) {

        double elapsed = current_time() - t0;
        if (elapsed > tr->config.max_time) {
            if (tr->config.verbose) {
                printf("  Terminating: max time exceeded (%.3f s).\n", elapsed);
            }
            break;
        }

        // gradient norm
        double gnorm = 0.0;
        if (tr->gradient) {
            for (size_t i = 0; i < n; ++i)
                gnorm += tr->gradient[i] * tr->gradient[i];
            gnorm = sqrt(gnorm);
        }

        if (tr->config.verbose) {
            printf("\nIter %zu:\n", iter);
            printf("  ||g||   = %.6e\n", gnorm);
            printf("  radius  = %.6e\n", tr->radius);
        }

        if (tr->gradient && gnorm <= tr->config.gtol) {
            if (tr->config.verbose) {
                printf("  Converged: ||g|| <= gtol (%.3e).\n", tr->config.gtol);
            }
            break;
        }

        // Call method-specific step: to update x_k, gradient, c_k, radius, etc. 
        if (tr->vptr->step) {
            double dummy = 0.0;
            int rc = tr->vptr->step(tr, &dummy);
            if (rc != 0) {
                if (tr->config.verbose) {
                    printf("  step() returned rc=%d; terminating.\n", rc);
                }
                break;
            }
        } else {
            if (tr->config.verbose) {
                printf("  No step() callback defined; terminating.\n");
            }
            break;
        }

        // for reporting fval, re-evaluate f at current iterate
        tr->vptr->fun(tr->x_k, &fx);
    }

    // fill result struct 
    res->iterations   = iter;
    res->final_radius = tr->radius;
    res->fval         = fx;
    res->time         = current_time() - t0;
    res->status       = 0;

    // copy solution back into x0 
    memcpy(x0, tr->x_k, n * sizeof(double));

    return 0;
}
