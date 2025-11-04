#include <stdlib.h>
#include <string.h>
#include "../include/opt_tr_powellyuan.h"
#include "../include/opt_tr.h"

// calculate your step and expected improvement
static int step(opt_tr *tr, double *expected_improvement)
{
    (void)tr;
    *expected_improvement = 1.0;
    return 0;
}

// calculate your updates trust-region radius
static int update_radius(opt_tr *tr, double ratio)
{
    
    (void)tr;
    (void)ratio;
    return 0;
}

// determine whether to accept a step or not
static int accept_step(opt_tr *tr, double ratio)
{
    (void)tr;
    (void)ratio;
    return 0;
}

// define your quasi-newton method
static int qn_update(opt_tr *tr)
{
    (void)tr;
    return 0;
}

// actually define the functions
static const opt_tr_vtable_t MY_VTABLE = {
    .step         = step,
    .update_radius= update_radius,
    .accept_step  = accept_step,
    .qn_update    = qn_update
};


// constructor
int tr_method_template_init(
    tr_method_template *solver,
    size_t n,
    opt_tr_fun_t fun,
    opt_tr_grad_t grad,
    opt_tr_hess_vec_t hess_vec,
    void *userdata
)
{
    if (!solver)
        return -1;

    memset(solver, 0, sizeof(*solver));

    
    int rc = opt_tr_init(&solver->base, n, fun, grad, hess_vec, userdata);
    if (rc != 0)
        return rc;

    // attach vtable
    solver->base.vptr = &MY_VTABLE;

    return 0;
}

// destructor
void tr_method_template_free(tr_method_template *solver)
{
    if (!solver)
        return;

    /* free base fields */
    opt_tr_free(&solver->base);
}
