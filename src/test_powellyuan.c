#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "../include/opt_tr.h"
#include "../include/opt_tr_powellyuan.h"

static void fun(const double *x, double *fx) {
    *fx = 0.5 * (0.5*x[0]*x[0]*x[0] + x[1]*x[1]);
}

static void grad(const double *x, double *g) {
    g[0] = x[0];
    g[1] = x[1];
}

// identity hessian
static void hess_vec(const double *x, const double *v, double *Hv) {
    (void)x;
    Hv[0] = v[0];
    Hv[1] = v[1];
}
static void constraints(const double *x, double *c) {
    c[0] = x[0] + x[1] - 1.0;
}
static void constr_jv(const double *x, const double *v, double *Jv) {
    (void)x;
    Jv[0] = v[0] + v[1];
}

int main() {

    const size_t m = 1;
    const size_t n = 2;

    double x_k[2]    = { 10.0, -100.0 };   // start location
    double test_x[2] = { 0.0,  0.0 };
    double gradient[2];
    double c_k[1];

    double radius = 1.0;

    opt_tr_options_t cfg = {0};
    cfg.max_iters   = 10;
    cfg.max_time    = 999999999;
    cfg.starting_radius = radius;
    cfg.radius_max  = 100.0;

    tr_method_powellyuan solver;

tr_method_powellyuan_init(
    &solver,
    m, n,
    x_k,
    test_x,
    gradient,
    radius,
    c_k,            
    cfg,
    fun,
    grad,
    hess_vec,
    constraints,
    constr_jv
);


    opt_tr *tr = &solver.base;

    double fx = 0.0;
    fun(tr->x_k, &fx);

    printf("Initial x=[%.6f %.6f], f=%.6f\n", tr->x_k[0], tr->x_k[1], fx);

    for (int k = 0; k < 30; ++k) {

        
        tr->vptr->grad(tr->x_k, tr->gradient);
        if (m > 0 && tr->vptr->constraints)
            tr->vptr->constraints(tr->x_k, tr->c_k);

        double gnorm = sqrt(tr->gradient[0]*tr->gradient[0] +
                            tr->gradient[1]*tr->gradient[1]);

        printf("\nIter %d:\n", k);
        printf("  x = [%.6f %.6f], ||g||=%.6f, c=%.6f\n",
               tr->x_k[0], tr->x_k[1], gnorm, tr->c_k[0]);

        if (gnorm < 1e-8 && fabs(tr->c_k[0]) < 1e-8) {
            printf("Terminated: converged.\n");
            break;
        }

        double predicted = 0.0;
        tr->vptr->step(tr, &predicted);

        double test_fx = 0.0;
        fun(tr->test_x, &test_fx);


        double actual_improvement    = fx - test_fx;
        double predicted_improvement = predicted;
        double ratio = 0.0;

        if (predicted_improvement > 0)
            ratio = actual_improvement / predicted_improvement;

        printf("  trial x = [%.6f %.6f], f=%.6f\n",
               tr->test_x[0], tr->test_x[1], test_fx);

        printf("  predicted=%.6e  actual=%.6e  ratio=%.6f\n",
               predicted_improvement, actual_improvement, ratio);

        tr->vptr->update_radius(tr, ratio);

        int accepted = tr->vptr->accept_step(tr, ratio);
        printf("  accepted=%d  new radius=%.6f\n", accepted, tr->radius);

        fun(tr->x_k, &fx);

        // quasi–Newton update (does nothing right now)
        tr->vptr->qn_update(tr);
    }

    printf("\nFinal x=[%.6f %.6f]\n", tr->x_k[0], tr->x_k[1]);

    tr_method_powellyuan_free(&solver);
    return 0;
}
