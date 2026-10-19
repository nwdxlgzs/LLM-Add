#include "add_chat.h"
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int fields=1, outcome=0;
    if(argc>1 && !strcmp(argv[argc-1],"--outcome")){outcome=1;argc--;}
    if(argc>1 && !strcmp(argv[argc-1],"--compact")){fields=0;argc--;}
    if (argc != 5 && argc != 7) { fprintf(stderr,"chat_eval MODEL COT REPORT_DIR NAME [A B] [--compact]\n"); return 2; }
    char error[256], path[1200], name[120];
    cn_model *m=cn_load(argv[1],error,sizeof(error));
    if(!m){fprintf(stderr,"%s\n",error);return 1;}
    if (argc == 7) {
        if (!add_mkdir(argv[3]) || !cn_model_inference_cache(m, 1)) return 1;
        char raw[1200], json[1200];
        snprintf(name,sizeof(name),"%s.txt",argv[4]);add_path(raw,sizeof(raw),argv[3],name);
        snprintf(name,sizeof(name),"%s.json",argv[4]);add_path(json,sizeof(json),argv[3],name);
        int ok=add_chat_predict_format(m,argv[5],argv[6],atoi(argv[2]),json,raw,fields);
        cn_model_free(m);return !ok;
    }
    add_chat_validation r;
    if(!add_mkdir(argv[3])||!(outcome ? add_chat_validate_outcome(m,atoi(argv[2]),&r)
                                   : add_chat_validate_format(m,atoi(argv[2]),&r,fields)))return 1;
    snprintf(name,sizeof(name),"%s-metrics.json",argv[4]);
    if(!add_path(path,sizeof(path),argv[3],name))return 1;
    FILE *f=fopen(path,"wb");if(!f)return 1;
    fprintf(f,"{\"correct\":%d,\"total\":200,\"validation_loss\":%.9g,\"tokens_correct\":%d,\"tokens_total\":%d,\"width_correct\":[%d,%d,%d,%d,%d],\"precision\":%d,\"parameters\":%zu,\"window\":%d",r.exact.correct,r.loss,r.token_correct,r.token_total,r.by_width[0],r.by_width[1],r.by_width[2],r.by_width[3],r.by_width[4],cn_model_precision(m),cn_parameter_count(m),cn_model_config(m)->window);
    const cn_config *c=cn_model_config(m);
    /* Exact widths make the published capacity comparison reproducible. */
    fprintf(f,",\"dim\":%d,\"ff\":%d,\"layers\":%d,\"head_dim\":%d,\"q_rank\":%d,\"o_rank\":%d,\"index_dim\":%d,\"engram_dim\":%d}\n",
            c->dim,c->ff_dim,c->layers,c->head_dim,c->q_rank,c->o_rank,c->index_dim,c->layer[1].engram.head_dim);
    fclose(f);
    const char *as[]={"0","9","99","1234567890","9999999999"};
    const char *bs[]={"0","9","1","9876543210","1"};
    for(int i=0;i<5;i++){
        char raw[1200], json[1200];
        snprintf(name,sizeof(name),"%s-sample-%d.txt",argv[4],i);add_path(raw,sizeof(raw),argv[3],name);
        snprintf(name,sizeof(name),"%s-sample-%d.json",argv[4],i);add_path(json,sizeof(json),argv[3],name);
        if(!add_chat_predict_format(m,as[i],bs[i],atoi(argv[2]),json,raw,fields))return 1;
    }
    cn_model_free(m);return 0;
}
