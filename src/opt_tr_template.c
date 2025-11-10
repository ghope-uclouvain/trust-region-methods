#include <stdlib.h>
#include <string.h>
#include "../include/opt_tr_template.h"
#include "../include/opt_tr.h"

static int step(opt_tr *tr, double *expected_improvement)
{
    (void)tr;
    *expected_improvement = 1.0;
    return 0;
}

static int update_radius(opt_tr *tr, double ratio)
{
    (void)tr;
    (void)ratio;
    return 0;
}

static int accept_step(opt_tr *tr, double ratio)
{
    (void)tr;
    (void)ratio;
    return 0;
}

static int qn_update(opt_tr *tr)
{
    (void)tr;
    return 0;
}

static const opt_tr_vtable_t MY_VTABLE = {
    .fun         = NULL,
    .grad        = NULL,
    .hess_vec    = NULL,
    .constraints = NULL,
    .constr_jv   = NULL,
    .step        = step,
    .update_radius = update_radius,
    .accept_step   = accept_step,
    .qn_update     = qn_update
};

int tr_method_template_init(
    tr_method_template *solver,
    size_t m,
    size_t n,
    double *x_k,
    double *test_k,
    double *gradient,
    double radius,
    double *c_k,
    opt_tr_options_t config,
    opt_tr_fun_t fun,
    opt_tr_grad_t grad,
    opt_tr_hess_vec_t hess_vec,
    opt_tr_constraints_t cons,
    opt_tr_constr_jv_t constr_jv
){
    if (!solver)
        return -1;

    memset(solver, 0, sizeof(*solver));

    opt_tr_vtable_t *vtable = malloc(sizeof(opt_tr_vtable_t));
    if (!vtable)
        return -1;

    *vtable = MY_VTABLE;
    vtable->fun         = fun;
    vtable->grad        = grad;
    vtable->hess_vec    = hess_vec;
    vtable->constraints = cons;
    vtable->constr_jv   = constr_jv;

    int rc = opt_tr_init(
        &solver->base,
        m,
        n,
        x_k,
        test_k,
        gradient,
        radius,
        c_k,
        config,
        vtable
    );

    if (rc != 0) {
        free(vtable);
        return rc;
    }

    return 0;
}

void tr_method_template_free(tr_method_template *solver)
{
    if (!solver)
        return;

    if (solver->base.vptr)
        free((void*)solver->base.vptr);

    opt_tr_free(&solver->base);
}
