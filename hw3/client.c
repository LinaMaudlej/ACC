#include <infiniband/verbs.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <string.h>
#include <assert.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "rpc_common.h"

struct ib_resources_t {
    struct ibv_context *context;
    struct ibv_pd *pd;
    struct ibv_qp *qp;
    struct ibv_cq *cq;
    int server_args_key;
    long long server_args_addr;
    int server_result_key;
    long long server_result_addr;
};

void rpc_call(struct ib_resources_t *ib_resources,
              int func_type,
              void *arg,
              int arg_size,
              void *result,
              int *result_size) {

    struct ibv_sge sg; /* scatter/gather element */
    struct ibv_send_wr wr; /* WQE */
    struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */
    struct ibv_wc wc; /* CQE */
    int ncqes; /* number of CQES polled */
    int wr_id = 0;

    /* step 1: send arg (if not NULL) to server using RDMA write */
    /* WQE needs to gather arg and write it to the server's (arg_key, arg_addr) */
    if (arg) {
        /* arg is not mapped to the HCA memory system, we need to map it using ibv_reg_mr */
        struct ibv_mr *mr = ibv_reg_mr(ib_resources->pd, arg, arg_size, 0);
        if (!mr) {
            printf("ERROR: ibv_reg_mr failed\n");
            exit(1);
        }

        /* now we'll perform a RDMA write to the argument buffer of the server.
         * RDMA write needs a gather element (local buffer) and a remote key & address (remote buffer) */

        /* gather element */
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)mr->addr;
        sg.length = arg_size;
        sg.lkey = mr->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = wr_id++; /* running index. should be inceremented for each WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */
        /* remote buffer parameters */
        wr.wr.rdma.remote_addr = ib_resources->server_args_addr;
        wr.wr.rdma.rkey = ib_resources->server_args_key;

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(ib_resources->qp, &wr, &bad_wr)) {
            printf("ERROR: ibv_post_send() failed\n");
            exit(1);
        }

        /* When WQE is completed we expect a CQE */
        /* wait for CQE on this operation so we know it is completed */
        do {
            ncqes = ibv_poll_cq(ib_resources->cq, 1, &wc);
        } while (ncqes == 0);
        if (ncqes < 0) {
            printf("ERROR: ibv_poll_cq() failed\n");
            exit(1);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            printf("ERROR: got CQE with error %d (line %d)\n", wc.status, __LINE__);
            exit(1);
        }
        assert(wc.opcode == IBV_WC_RDMA_WRITE); /* make sure it's the CQE we're expecting */
    }

    /* step 2: send RDMA write with immediate (with no data) to tell server it has work to do.
     *         we'll use the immediate value as the ID of the function we want it to perform
     *         immediate value is a 32-bit value the sender sets in the write/send with immediate WQE
     *         and receiver gets it in its CQE */
    memset(&wr, 0, sizeof(struct ibv_send_wr));
    wr.wr_id = wr_id++; /* running index. should be inceremented for each WQE */
    wr.sg_list = NULL; /* we don't want to send any data */
    wr.num_sge = 0;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;/* TODO: fill in this. refer to the man page or to the references mentioned in the homework PDF */
    wr.imm_data = func_type; /* immediate data will contain the function that we want to perform */
    wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this assignment. generates CQE */
    wr.wr.rdma.remote_addr = ib_resources->server_args_addr; /* we'll "write" to argument buffer, but we're not writing anything really */
    wr.wr.rdma.rkey = ib_resources->server_args_key;


    /* TODO: post the WQE using ibv_post_send() */
    if (ibv_post_send(ib_resources->qp, &wr, &bad_wr)) {
	printf("ERROR: ibv_post_send() failed\n");
        exit(1);
    }

    /* TODO: wait for completion of this WQE using ibv_poll_cq() */
    /* wait for CQE on this operation so we know it is completed.
     * make sure ibv_poll_cq() succeeded and cqe is not error*/
        do {
            ncqes = ibv_poll_cq(ib_resources->cq, 1, &wc);
        } while (ncqes == 0);
        if (ncqes < 0) {
            printf("ERROR: ibv_poll_cq() failed\n");
            exit(1);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            printf("ERROR: got CQE with error %d (line %d)\n", wc.status, __LINE__);
            exit(1);
        }

        assert(wc.opcode == IBV_WC_RDMA_WRITE); /* make sure it's the CQE we're expecting */

    /* ok now we want to know if server has finished to produce the result */
    if (result) {
        /* TODO: wait for CQE telling us that result is ready in the server using ibv_poll_cq()
         * make sure ibv_poll_cq() succeeded and CQE is not error */
	do {
            ncqes = ibv_poll_cq(ib_resources->cq, 1, &wc);
        } while (ncqes == 0);
        if (ncqes < 0) {
            printf("ERROR: ibv_poll_cq() failed\n");
            exit(1);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            printf("ERROR: got CQE with error %d (line %d)\n", wc.status, __LINE__);
            exit(1);
        }
        assert(wc.opcode == IBV_WC_RECV);
        *result_size = wc.imm_data;

        /* now we'll read the result from the server's result buffer.
         * we'll read it into our own result buffer, but we need first to register it using ibv_reg_mr() */
        struct ibv_mr *mr = ibv_reg_mr(ib_resources->pd, result ,*result_size /*TODO fill missing arguments */, IBV_ACCESS_LOCAL_WRITE);
        if (!mr) {
            printf("ERROR: ibv_reg_mr failed\n");
            exit(1);
        }

        /* perform RDMA read operation of the result
         * remember: we will read from remote address ib_resources->server_result_addr with key ib_resources->server_result_key */
        memset(&sg, 0, sizeof(struct ibv_sge));
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        /* gather item: tells where the data should be written, we'll use the memory region parameters here */

        sg.addr = (uintptr_t)mr->addr; /* TODO fill this*/
        sg.length = *result_size; /* TODO fill this */
        sg.lkey = mr->lkey; /* TODO fill this */

        wr.wr_id = wr_id++;
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_READ; /* TODO fill this */
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = ib_resources->server_result_addr; /*TODO complete*/;
        wr.wr.rdma.rkey = ib_resources->server_result_key; /*TODO complete*/;

        /* TODO post the WQE for execution using ibv_post_send() */
	if (ibv_post_send(ib_resources->qp, &wr, &bad_wr)) {
            printf("ERROR: ibv_post_send() failed\n");
            exit(1);
        }

        /* wait for CQE on this operation so we know it is completed */
        do {
            ncqes = ibv_poll_cq(ib_resources->cq, 1, &wc);
        } while (ncqes == 0);
        if (ncqes < 0) {
            printf("ERROR: ibv_poll_cq() failed\n");
            exit(1);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            printf("ERROR: got CQE with error %d (line %d)\n", wc.status, __LINE__);
            exit(1);
        }
        assert(wc.opcode == IBV_WC_RDMA_READ);

        /* done, we have the result and we're finished */

        ibv_dereg_mr(mr); /* and now we deregister the local result buffer because we finished using it */
    }

}

void rpc_decrypt(struct ib_resources_t *ib_resources,
                 char *str) {
    char result[1000];
    int result_size;
    rpc_call(ib_resources,
             RPC_FUNC_DECRYPT,
             str,
             strlen(str) + 1,
             result,
             &result_size);

    printf("%s", result);
}

void rpc_collatz(struct ib_resources_t *ib_resources,
                 int num) {
    int result[1000];
    int result_size;
    rpc_call(ib_resources,
            RPC_FUNC_COLLATZ,
            (char *)&num,
            sizeof(int),
            (char *)result,
            &result_size);
    int nitems = result_size / sizeof(int);
    printf("collatz chain of %d: ", num);
    int i;
    for (i = 0; i < nitems; i++) {
        printf("%d%s", result[i], (i == nitems - 1) ? "\n" : ", ");
    }
}

void rpc_done(struct ib_resources_t *ib_resources) {
    rpc_call(ib_resources,
            RPC_FUNC_DONE,
            NULL,
            0,
            NULL,
            NULL);
    printf("done\n");
}

struct ib_resources_t *setup_connection(int tcp_port) {
    /* first we'll connect to server via a TCP socket to exchange Infiniband parameters */
    int sfd;
    sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) {
        perror("socket");
        exit(1);
    }

    struct sockaddr_in server_addr;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); /* server is on same machine */
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(tcp_port);

    if (connect(sfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        perror("connect");
        exit(1);
    }

    /*ok let's start creating infiniband resources*/
    struct ib_resources_t *ib_resources = malloc(sizeof(struct ib_resources_t));

    /* get device list */
    struct ibv_device **device_list = ibv_get_device_list(NULL);
    if (!device_list) {
        printf("ERROR: ibv_get_device_list failed\n");
        exit(1);
    }

    /* select first (and only) device to work with */
    struct ibv_context *context = ibv_open_device(device_list[0]);

    /* create protection domain (PD) */
    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        printf("ERROR: ibv_alloc_pd() failed\n");
        exit(1);
    }

    /* create completion queue (CQ). We'll use same CQ for both send and receive parts of the QP */
    struct ibv_cq *cq = ibv_create_cq(context, 100, NULL, NULL, 0); /* create a CQ with place for 100 CQEs */
    if (!cq) {
        printf("ERROR: ibv_create_cq() failed\n");
        exit(1);
    }

    /* create QP */
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.qp_type = IBV_QPT_RC; /* we'll use RC transport service, which supports RDMA */
    qp_init_attr.cap.max_send_wr = 1; /* max of 1 WQE in-flight in SQ. that's enough for us */
    qp_init_attr.cap.max_recv_wr = MAX_RECV_WQES; /* max of 8 WQE's in-flight in RQ. that's more than enough for us */
    qp_init_attr.cap.max_send_sge = 1; /* 1 SGE in each send WQE */
    qp_init_attr.cap.max_recv_sge = 1; /* 1 SGE in each recv WQE */
    struct ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
        printf("ERROR: ibv_create_qp() failed\n");
        exit(1);
    }

    /* get server info (QP number and LID) */
    struct ib_info_t server_info;
    int ret;
    ret = recv(sfd, &server_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("recv");
        exit(1);
    }

    /* send our info to server */
    /*get our LID first */
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(context, IB_PORT_CLIENT, &port_attr);
    if (ret) {
        printf("ERROR: ibv_query_port() failed\n");
        exit(1);
    }
    struct ib_info_t my_info;
    my_info.lid = port_attr.lid;
    my_info.qpn = qp->qp_num;
    ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("send");
        exit(1);
    }
    /* we don't need TCP anymore. kill the socket */
    close(sfd);

    /* now need to connect the QP to the server''s QP.
     * this is a multi-phase process, moving the state machine of the QP step by step
     * until we are ready */
    struct ibv_qp_attr qp_attr;

    /*QP state: RESET -> INIT */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = IB_PORT_CLIENT;
    qp_attr.qp_access_flags = 0; /* server doesn't need RDMA access to our QP */
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to INIT failed\n");
        exit(1);
    }


    /*QP: state: INIT -> RTR (Ready to Receive) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = server_info.qpn; /* qp number of server */
    qp_attr.rq_psn      = 0 ;
    qp_attr.max_dest_rd_atomic = 1; /* max in-flight RDMA reads */
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 0; /* No Network Layer (L3) */
    qp_attr.ah_attr.dlid = server_info.lid; /* LID (L2 Address) of server */
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = IB_PORT_CLIENT;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU| IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to RTR failed\n");
        exit(1);
    }

    /*QP: state: RTR -> RTS (Ready to Send) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.max_rd_atomic = 1;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to RTS failed\n");
        exit(1);
    }

    /* now let's populate the receive QP with recv WQEs */
    struct ibv_recv_wr recv_wr; /* this is the receive work request (the verb's representation for receive WQE) */
    int i;
    for (i = 0; i < MAX_RECV_WQES; i++) {
        memset(&recv_wr, 0, sizeof(struct ibv_recv_wr));
        recv_wr.wr_id = i;
        recv_wr.sg_list = NULL;
        recv_wr.num_sge = 0;
        if (ibv_post_recv(qp, &recv_wr, NULL)) {
            printf("ERROR: ibv_post_recv() failed\n");
            exit(1);
        }
    }

    ib_resources->context = context;
    ib_resources->pd = pd;
    ib_resources->qp = qp;
    ib_resources->cq = cq;
    ib_resources->server_args_key = server_info.mkey_args;
    ib_resources->server_args_addr = server_info.addr_args;
    ib_resources->server_result_key = server_info.mkey_result;
    ib_resources->server_result_addr = server_info.addr_result;

    return ib_resources;
}

void teardown_connection(struct ib_resources_t *ib_resources) {
    ibv_destroy_qp(ib_resources->qp);
    ibv_destroy_cq(ib_resources->cq);
    ibv_dealloc_pd(ib_resources->pd);
    ibv_close_device(ib_resources->context);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("usage: %s <tcp port>\n", argv[0]);
        exit(1);
    }
    int tcp_port = atoi(argv[1]);

    struct ib_resources_t *ib_resources = setup_connection(tcp_port);

    /* let's test our RPC */
    printf("ok let's try some decryption first.\n");
    char str[1000];
    strcpy(str, "Frrzf ");
    rpc_decrypt(ib_resources, str);

    strcpy(str, "gb ");
    rpc_decrypt(ib_resources, str);

    strcpy(str, "jbex!\n");
    rpc_decrypt(ib_resources, str);

    printf("now let's do some collatz chains\n");
    rpc_collatz(ib_resources, 1);
    rpc_collatz(ib_resources, 2);
    rpc_collatz(ib_resources, 3);
    rpc_collatz(ib_resources, 4);
    rpc_collatz(ib_resources, 5);
    rpc_collatz(ib_resources, 6);
    rpc_collatz(ib_resources, 7);
    rpc_collatz(ib_resources, 8);
    rpc_collatz(ib_resources, 9);

    rpc_done(ib_resources);

    teardown_connection(ib_resources);

    return 0;
}
