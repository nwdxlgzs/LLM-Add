/* Publish actual whole-dialogue stage selections without optimizer buffers. */
#include "add_chat.h"
#include <stdlib.h>
int main(int argc, char **argv) {
    if ((argc != 3 && argc != 4 && argc != 5) || !add_mkdir(argv[2])) return 2;
    char src[1200], dst[1200], name[120], error[256];
    for (int i = 1; i <= (argc>4?atoi(argv[4]):6); i++) {
        snprintf(name, sizeof(name), "stage-%d.cn", i);
        if (!add_path(src, sizeof(src), argv[1], name)) return 1;
        cn_model *m = cn_load(src, error, sizeof(error));
        if (!m) { fprintf(stderr, "%s\n", error); return 1; }
        snprintf(name, sizeof(name), "%s-stage-%d.cn", argc>=4?argv[3]:"dialogue-v4", i);
        if (!add_path(dst, sizeof(dst), argv[2], name) || !cn_save(m, dst, 0)) return 1;
        cn_model_free(m);
        if(i==7) {
            const char *variants[]={"entry","last"};
            for(int v=0;v<2;v++) {
                snprintf(name,sizeof(name),"stage-7-%s.cn",variants[v]);
                if(!add_path(src,sizeof(src),argv[1],name))return 1;
                m=cn_load(src,error,sizeof(error));if(!m)return 1;
                snprintf(name,sizeof(name),"%s-stage-7-%s.cn",argc>=4?argv[3]:"dialogue-v6",variants[v]);
                if(!add_path(dst,sizeof(dst),argv[2],name)||!cn_save(m,dst,0))return 1;
                cn_model_free(m);
            }
        }
    }
    return 0;
}
