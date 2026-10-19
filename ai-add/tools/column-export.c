#include "add_column.h"
#include <stdlib.h>
int main(int argc,char **argv){
    if(argc!=3)return 2;
    if(!add_mkdir(argv[2]))return 1;
    char src[1200],dst[1200],name[100],error[256];
    for(int stage=1;stage<=6;stage++)for(int variant=0;variant<3;variant++){
        const char *suffix=variant==0?"":variant==1?"-last":"-entry";
        snprintf(name,sizeof(name),"stage-%d%s.cn",stage,suffix);if(!add_path(src,sizeof(src),argv[1],name))return 1;
        cn_model *m=cn_load(src,error,sizeof(error));if(!m)return 1;
        snprintf(name,sizeof(name),"column-v2-stage-%d%s.cn",stage,suffix);if(!add_path(dst,sizeof(dst),argv[2],name)||!cn_save(m,dst,0))return 1;
        cn_model_free(m);
    }
    for(int precision=0;precision<=2;precision++){
        add_path(src,sizeof(src),argv[1],precision==2?"stage-6-last.cn":"stage-5.cn");
        cn_model *m=cn_load(src,error,sizeof(error));if(!m)return 1;
        if(precision<2&&!cn_model_quantize(m,(cn_dtype)precision))return 1;
        snprintf(name,sizeof(name),"column-v2-final%s.cn",precision==0?"":precision==1?"-fp8":"-fp4");
        if(!add_path(dst,sizeof(dst),argv[2],name)||!cn_save(m,dst,0))return 1;
        add_chat_validation r;if(!add_column_validate(m,&r))return 1;
        printf("%s %d/200 margin=%g bytes stored as precision %d\n",name,r.exact.correct,r.exact.margin,cn_model_precision(m));
        cn_model_free(m);
    }
    return 0;
}
