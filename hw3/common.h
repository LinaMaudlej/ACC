#ifndef __RPC_COMMON_H__
#define __RPC_COMMON_H__

struct ib_info_t {
    int lid;
    int qpn;
    int mkey_args;
    long long addr_args;
    int mkey_result;
    long long addr_result;
};

enum {RPC_FUNC_DECRYPT = 1, RPC_FUNC_COLLATZ = 2, RPC_FUNC_DONE = 0xff};

#define IB_PORT_SERVER 1
#define IB_PORT_CLIENT 2
#define MAX_RECV_WQES 128

#endif
