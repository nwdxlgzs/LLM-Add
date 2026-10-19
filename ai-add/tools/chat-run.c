/* Native driver for reproducing the browser's actual chat training recipe. */
#include "add_chat.h"
#include "add_session.h"
#include "add_column.h"
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 9) {
        fprintf(stderr, "chat_run DIR COT PRETRAIN SFT1 SFT2 SFT3 RL QAT [BATCH=4] [MODEL|new] [FIRST=1] [PROTOCOL=3] [QAT_BATCH=BATCH] [DIM=24] [FF=DIM] [LAYERS=6] [OUTCOME_RL=0] [CURRICULUM_COMPLETED=0]\n");
        return 2;
    }
    cn_config c;
    add_chat_config(&c, 20261018);
    c.max_seq = ADD_CHAT_CAPACITY;
    int dim = argc > 14 ? atoi(argv[14]) : 24, ff = argc > 15 ? atoi(argv[15]) : dim;
    if (!add_chat_architecture(&c, argc>16?atoi(argv[16]):6, dim, ff)) { fprintf(stderr, "Invalid architecture or >500000 parameters\n"); return 2; }
    add_session *s = add_session_create(&c, argv[1]);
    int cot = atoi(argv[2]), batch = argc > 9 ? atoi(argv[9]) : 4;
    int protocol = argc > 12 ? atoi(argv[12]) : 3;
    if (!s || (argc > 10 && strcmp(argv[10], "new") && !add_session_load(s, argv[10])) ||
        !add_session_set_chat(s, protocol) || !add_session_set_cot(s, cot))
        return 1;
    const cn_config *actual = cn_model_config(add_session_model(s));
    printf("parameters=%zu layers=%d dim=%d ff=%d head_dim=%d q_rank=%d o_rank=%d index_dim=%d engram_dim=%d\n",
           cn_parameter_count(add_session_model(s)), actual->layers, actual->dim, actual->ff_dim,
           actual->head_dim, actual->q_rank, actual->o_rank, actual->index_dim,
           actual->layer[1].engram.head_dim);
    fflush(stdout);
    char path[1200];
    add_path(path, sizeof(path), argv[1], "progress.csv");
    int first = argc > 11 ? atoi(argv[11]) : 1;
    FILE *log = fopen(path, first>1?"ab":"wb");
    if (!log) return 1;
    if(first==1) fputs("stage,step,loss,correct,reward,informative,seconds,validation_loss,w1,w2,w4,w6,w10,best_step,last_correct\n", log);
    for (int stage = first; stage <= 7; stage++) {
        int steps = stage==7 ? (argc>17?atoi(argv[17]):0) : atoi(argv[stage + 2]);
        if (!steps) continue;
        if (!add_session_begin(s, stage, steps, stage == 6 && argc > 13 ? atoi(argv[13]) : batch, 0)) return 1;
        if(stage==4 && argc>18 && atoi(argv[18])>0 &&
           !add_session_continue_curriculum(s,atoi(argv[18])))return 1;
        int result, metric = -2;
        do {
            result = add_session_tick(s);
            if (result < 0) return 1;
            const add_progress *p = add_session_progress(s);
            if (p->metric_step != metric || result) {
                metric = p->metric_step;
                fprintf(log, "%d,%d,%.8g,%d,%.8g,%d,%.3f,%.8g,%d,%d,%d,%d,%d,%d,%d\n",stage,p->step,p->loss,p->correct,p->reward,p->informative,p->seconds,p->validation_loss,p->by_width[0],p->by_width[1],p->by_width[2],p->by_width[3],p->by_width[4],p->best_step,p->last_correct);
                fflush(log);
                if(stage==5 || stage==7) {
                    char tracepath[1200],tracename[80];
                    snprintf(tracename,sizeof(tracename),"stage-%d-rollouts-%d.json",stage,p->step);
                    if(!add_path(tracepath,sizeof(tracepath),argv[1],tracename) ||
                       !add_chat_save_rollouts(add_session_rollouts(s),tracepath))return 1;
                }
                printf("stage=%d step=%d loss=%.5f complete=%d/200 widths=%d,%d,%d,%d,%d valCE=%.4f best=%d last=%d informative=%d seconds=%.1f\n",stage,p->step,p->loss,p->correct,p->by_width[0],p->by_width[1],p->by_width[2],p->by_width[3],p->by_width[4],p->validation_loss,p->best_step,p->last_correct,p->informative,p->seconds);
                fflush(stdout);
            }
        } while (!result);
        const char *as[] = {"0", "9", "99", "1234567890", "9999999999"};
        const char *bs[] = {"0", "9", "1", "9876543210", "1"};
        for (int i = 0; i < 5; i++) {
            char raw[1200], json[1200], name[80];
            snprintf(name,sizeof(name),"stage-%d-sample-%d.txt",stage,i);
            add_path(raw,sizeof(raw),argv[1],name);
            snprintf(name,sizeof(name),"stage-%d-sample-%d.json",stage,i);
            add_path(json,sizeof(json),argv[1],name);
            if (!(protocol == 2 && cot ? add_column_predict(add_session_model(s),as[i],bs[i],json,raw)
                                      : add_chat_predict_format(add_session_model(s),as[i],bs[i],cot,json,raw,protocol==3))) return 1;
        }
    }
    fclose(log);
    add_session_free(s);
    return 0;
}
