#ifndef OPT_TR_POWELLYUAN
#define OPT_TR_POWELLYUAN

#include "opt_tr.h"

typedef struct {
    opt_tr base;
} tr_method_template;

int tr_method_powellyuan_init(
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
);

void tr_method_powellyuan_free(tr_method_template *solver);

#endif
