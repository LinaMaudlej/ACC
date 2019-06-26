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

#define TCP_PORT_OFFSET 23456
#define TCP_PORT_RANGE 1000
#define ARG_BUF_SIZE (1024 * 1024)
#define RES_BUF_SIZE (1024 * 1024)

int do_decrypt(char *arg_buf, char *result_buf) {
    int i;
    for (i = 0; arg_buf[i] != '\0'; i++) {
        char c_in = arg_buf[i];
        char c_out = c_in;

        if ((c_in >= 'a') && (c_in <= 'z')) {
            c_out = 'a' + (((c_in - 'a') + 13) % 26);
        } else if ((c_in >= 'A') && (c_in <= 'Z')) {
            c_out = 'A' + (((c_in - 'A') + 13) % 26);
        }
        result_buf[i] = c_out;
    }
    result_buf[i] = '\0';
    return (i + 1);
}

int do_collatz(char *arg_buf, char *result_buf) {
    int num = *(int *)arg_buf;
    int *res = (int *)result_buf;

    int i;
    for (i = 0; ; i++) {
        res[i] = num;
        if (num == 1) break;
        if (num % 2 == 0) num /= 2;
        else              num = 3 * num + 1;
    }
    return sizeof(int) * (i + 1);
}

int main(int argc, char *argv[]) {
    /* setup a TCP connection for initial negotiation with client */
    int lfd, sfd;
    lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) {
        perror("socket");
        exit(1);
    }


    int tcp_port;
    if (argc < 2) {
        srand(time(NULL));
        tcp_port = TCP_PORT_OFFSET + (rand() % TCP_PORT_RANGE); /* to avoid conflicts with other users of the machine */
    } else {
        tcp_port = atoi(argv[1]);
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(tcp_port);

    if (bind(lfd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in)) < 0) {
        perror("bind");
        exit(1);
    }

    listen(lfd, 1);

    printf("Server waiting on port %d. Client can connect\n", tcp_port);
    sfd = accept(lfd, NULL, NULL);
    if (sfd < 0) {
        perror("accept");
        exit(1);
    }
    printf("client connected\n");

    /* now that client has connected to us via TCP we'll open up some Infiniband resources and send it the parameters */

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

    /* allocate a memory region for the RPC arguments.
     * must be writeable by client */
    struct ibv_mr *mr_args;
    char *arg_buf = malloc(ARG_BUF_SIZE);
    mr_args = ibv_reg_mr(pd, arg_buf, ARG_BUF_SIZE, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_args) {
        printf("ibv_reg_mr() failed for argument buffer\n");
        exit(1);
    }

    /* allocate a memory region for the RPC result.
     * must be readable by client */
    struct ibv_mr *mr_result;
    char *result_buf = malloc(RES_BUF_SIZE);
    mr_result = ibv_reg_mr(pd, result_buf, RES_BUF_SIZE, IBV_ACCESS_REMOTE_READ);
    if (!mr_result) {
        printf("ibv_reg_mr() failed for result buffer\n");
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

    /* ok, before we continue we need to get info about the client' QP, and send it info about ours.
     * namely: QP number, and LID.
     * we'll use the TCP socket for that */

    /* first query port for its LID (L2 address) */
    int ret;
    struct ibv_port_attr port_attr;
    ret = ibv_query_port(context, IB_PORT_SERVER, &port_attr);
    if (ret) {
        printf("ERROR: ibv_query_port() failed\n");
        exit(1);
    }

    /* now send our info to client */
    struct ib_info_t my_info;
    my_info.lid = port_attr.lid;
    my_info.qpn = qp->qp_num;
    my_info.mkey_args = mr_args->rkey;
    my_info.addr_args = (uintptr_t)mr_args->addr;
    my_info.mkey_result = mr_result->rkey;
    my_info.addr_result = (uintptr_t)mr_result->addr;
    ret = send(sfd, &my_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("send");
        exit(1);
    }

    /* get client's info */
    struct ib_info_t client_info;
    recv(sfd, &client_info, sizeof(struct ib_info_t), 0);
    if (ret < 0) {
        perror("recv");
        exit(1);
    }

    /* we don't need TCP anymore. kill the socket */
    close(sfd);
    close(lfd);


    /* now need to connect the QP to the client's QP.
     * this is a multi-phase process, moving the state machine of the QP step by step
     * until we are ready */
    struct ibv_qp_attr qp_attr;

    /*QP state: RESET -> INIT */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = IB_PORT_SERVER;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ; /* we'll allow client to RDMA write and read on this QP */
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        printf("ERROR: ibv_modify_qp() to INIT failed\n");
        exit(1);
    }


    /*QP: state: INIT -> RTR (Ready to Receive) */
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = client_info.qpn; /* qp number of client */
    qp_attr.rq_psn      = 0 ;
    qp_attr.max_dest_rd_atomic = 1; /* max in-flight RDMA reads */
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.is_global = 0; /* No Network Layer (L3) */
    qp_attr.ah_attr.dlid = client_info.lid; /* LID (L2 Address) of client */
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = IB_PORT_SERVER;
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
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

    /* now finally we get to the actual work */
    /* so the protocol goes like this:
     * 1. we'll wait for a CQE indicating that we got an RDMA Write with Immediate from the client.
     *    this tells us we have new work to do.
     *    The immediate value (written in the CQE) will tell us what function we will be performing
     * 2. now we know that the arguments are in arg_buf.
     *    so we perform the function (from 1) on the arguments (from 2).
     * 3. we store the result in result_buf
     * 4. we send an RDMA send with immediate to the client (with no data),
     *    telling it that the result is ready for reading.
     * 5. we go back to 1 to perform another operation
     */

    for (i = 0; ; i++) {
        /*step 1: poll for CQE */
        struct ibv_wc wc;
        int ncqes;
        do {
            ncqes = ibv_poll_cq(cq, 1, &wc);
        } while (ncqes == 0);
        if (ncqes < 0) {
            printf("ERROR: ibv_poll_cq() failed\n");
            exit(1);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            printf("ERROR: got CQE with error %d (line %d)\n", wc.status, __LINE__);
            exit(1);
        }

        assert(wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM);
        printf("Got a CQE :)\n");

        /* perform RPC and store result in result_buf */
        int res_size = 0;
        if (wc.imm_data == RPC_FUNC_DONE) {
            printf("client says it's done. we'll exit..\n");
            break;
        } else if (wc.imm_data == RPC_FUNC_DECRYPT) {
            printf("will perform function: decrypt\n");
            res_size = do_decrypt(arg_buf, result_buf);
        } else if (wc.imm_data == RPC_FUNC_COLLATZ) {
            printf("will perform function: collatz\n");
            res_size = do_collatz(arg_buf, result_buf);
        } else {
            printf("unexpected immediate value %d\n", wc.imm_data);
            exit(1);
        }

        /* send RDMA send with immediate to client to tell it we're done */
        struct ibv_send_wr send_wr;
        struct ibv_send_wr *bad_send_wr;
        memset(&send_wr, 0, sizeof(struct ibv_send_wr));
        send_wr.wr_id = i;
        send_wr.sg_list = NULL;
        send_wr.num_sge = 0;
        send_wr.opcode = IBV_WR_SEND_WITH_IMM;
        send_wr.send_flags = IBV_SEND_SIGNALED;
        send_wr.imm_data = res_size;

        if (ibv_post_send(qp, &send_wr, &bad_send_wr)) {
            printf("ERROR: ibv_post_send() failed\n");
            exit(1);
        }

        /* now poll CQ for completion of our RDMA send with immediate.*/
        do {
            ncqes = ibv_poll_cq(cq, 1, &wc);
        } while (ncqes == 0);
        if (ncqes < 0) {
            printf("ERROR: ibv_poll_cq() failed\n");
            exit(1);
        }
        if (wc.status != IBV_WC_SUCCESS) {
            printf("ERROR: got CQE with error %d (line %d)\n", wc.status, __LINE__);
            exit(1);
        }
        assert(wc.opcode == IBV_WC_SEND);

        /* we notified client. now it should read the result and then we go again */
    }



    printf("Done\n");
    /* cleanup */
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dereg_mr(mr_args);
    ibv_dereg_mr(mr_result);
    ibv_dealloc_pd(pd);
    ibv_close_device(context);
    return 0;
}
