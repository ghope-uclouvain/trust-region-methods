#ifndef OPT_TR_POWELLYUAN
#define OPT_TR_POWELLYUAN

#include "opt_tr.h"

typedef struct {
    opt_tr base;
} tr_method_template;

// constructor
int opt_tr_powellyuan_init(
    tr_method_template *solver,
    size_t n,                   
    opt_tr_fun_t fun,
    opt_tr_grad_t grad,
    opt_tr_hess_vec_t hess_vec,
    void *userdata
);

// destructor
void opt_tr_powellyuan_free(tr_method_template *solver);

#endif
