/* Capacity pilots use the exact dialogue objective and CPU engine. */
#include "add_chat.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) { fprintf(stderr, "chat_bench DIM FF UPDATES [LAYERS=6]\n"); return 2; }
    cn_config c;
    add_chat_config(&c, 20261018);
    if (!add_chat_architecture(&c, argc>4?atoi(argv[4]):6, atoi(argv[1]), atoi(argv[2]))) return 2;
    char error[256];
    cn_model *m = cn_model_create(&c, error, sizeof(error));
    if (!m) { fprintf(stderr, "%s\n", error); return 1; }
    const char *a[] = {"7", "90", "7140345462"}, *b[] = {"8", "70", "1095133250"};
    cn_optimizer opt = cn_optimizer_default();
    opt.learning_rate = .001f;
    int updates = atoi(argv[3]);
    if (updates < 1) return 2;
    printf("{\"layers\":%d,\"dim\":%d,\"ff\":%d,\"parameters\":%zu,\"head_dim\":%d,\"q_rank\":%d,\"o_rank\":%d,\"index_dim\":%d,\"engram_dim\":%d,\"measurements\":[",
           c.layers,c.dim,c.ff_dim,cn_parameter_count(m),c.head_dim,c.q_rank,c.o_rank,c.index_dim,c.layer[1].engram.head_dim);
    for (int w = 0; w < 3; w++) {
        int t[ADD_CHAT_CAPACITY], prompt;
        int n = add_chat_example(a[w],b[w],1,0,t,ADD_CHAT_CAPACITY,&prompt);
        cn_token_objective objective[ADD_CHAT_CAPACITY] = {{0}};
        float weight = 0;
        for (int j = prompt; j < n; j++) weight += t[j]>='0'&&t[j]<='9'?3:1;
        for (int j = prompt; j < n; j++) objective[j].ce=(t[j]>='0'&&t[j]<='9'?3:1)/weight;
        cn_metrics stats = {0};
        double start = add_now();
        for (int i = 0; i < updates; i++)
            if (!cn_objective_backward(m,t,(size_t)n,objective,0,1,&stats) ||
                !cn_optimizer_step(m,&opt,&stats) || !isfinite(stats.loss)) return 1;
        printf("%s{\"digits\":%zu,\"tokens\":%d,\"updates\":%d,\"seconds_per_sequence\":%.9g,\"arena_bytes\":%zu}",
               w?",":"",strlen(a[w]),n,updates,(add_now()-start)/updates,stats.arena_bytes);
        fflush(stdout);
    }
    puts("]}");
    cn_model_free(m);
    return 0;
}
