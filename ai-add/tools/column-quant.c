#include "add_column.h"
#include <math.h>
#include <stdlib.h>
int main(int argc,char **argv){
    if(argc!=7){fprintf(stderr,"column_quant INPUT OUTPUT PRECISION STEPS BATCH LR\n");return 2;}
    char error[256],last[1200];cn_model*m=cn_load(argv[1],error,sizeof(error));
    int precision=atoi(argv[3]),steps=atoi(argv[4]),batch=atoi(argv[5]);
    if(!m||precision<1||precision>2||steps<1||batch<1||batch>32||!cn_model_set_qat(m,(cn_dtype)precision)||!cn_optimizer_reset(m))return 1;
    cn_optimizer opt=cn_optimizer_default();opt.learning_rate=strtof(argv[6],NULL);opt.beta2=.99f;opt.weight_decay=0;
    if(!isfinite(opt.learning_rate)||opt.learning_rate<=0||opt.learning_rate>.01f)return 2;
    uint64_t rng=20261023;int best=-1;float margin=-1e9f;
    for(int i=1;i<=steps;i++){
        cn_metrics stats;double reward;int informative;
        if(!add_column_update(m,NULL,6,i,steps,batch,&rng,&opt,&stats,&reward,&informative))return 1;
        if(i%25==0||i==steps){
            add_chat_validation r;if(!add_column_validate(m,&r))return 1;
            printf("step=%d correct=%d/200 min_margin=%g CE=%g\n",i,r.exact.correct,r.exact.margin,r.loss);fflush(stdout);
            if(r.exact.correct>best||(r.exact.correct==best&&r.exact.margin>margin)){
                best=r.exact.correct;margin=r.exact.margin;if(!cn_save(m,argv[2],0))return 1;
            }
            if(best==200&&margin>=2)break;
        }
    }
    snprintf(last,sizeof(last),"%s-last.cn",argv[2]);int ok=cn_save(m,last,1);cn_model_free(m);return ok&&best==200?0:1;
}
