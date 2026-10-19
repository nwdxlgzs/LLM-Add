#include "add_chat.h"
#include "add_session.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "chat FAIL %d: %s\n", __LINE__, #x);                                   \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
static int count(const int *t, int n, int id) {
    int k = 0;
    for (int i = 0; i < n; i++)
        k += t[i] == id;
    return k;
}
static int same_file(const char *a, const char *b) {
    FILE *fa = fopen(a,"rb"), *fb = fopen(b,"rb");
    if (!fa || !fb) { if(fa)fclose(fa); if(fb)fclose(fb); return 0; }
    int x,y,ok=1;
    do { x=fgetc(fa);y=fgetc(fb);if(x!=y){ok=0;break;} } while(x!=EOF);
    fclose(fa);fclose(fb);return ok;
}
int main(int argc, char **argv) {
    CHECK(argc == 2);
    int t[ADD_CHAT_CAPACITY], prompt;
    int n = add_chat_example("99", "1", 1, 0, t, ADD_CHAT_CAPACITY, &prompt);
    CHECK(n > prompt && count(t, n, CN_THINK) == 1 && count(t, n, CN_THINK_END) == 1);
    CHECK(count(t, prompt, CN_THINK) == 0 && count(t, prompt, CN_THINK_END) == 0);
    CHECK(count(t, n, CN_IM_START) == 2 && count(t, n, CN_IM_END) == 2);
    CHECK(t[n - 1] == CN_EOT && t[n - 2] == CN_IM_END);
    char text[2048] = {0};
    size_t bytes = cn_decode(t, (size_t)n, 1, text, sizeof(text) - 1);
    CHECK(bytes < sizeof(text));
    CHECK(!strcmp(text, "<|im_start|>user\n99+1<|im_end|>\n<|im_start|>assistant\n<think>\n"
                        "pos=0 a=9 b=1 cin=0 -> 0 1\n"
                        "pos=1 a=9 b=0 cin=1 -> 0 1\n"
                        "</think>\n100<|im_end|><|endoftext|>"));
    CHECK(add_chat_reward("99","1",1,t+prompt,n-prompt,1)==1);
    n = add_chat_example_format("99","1",1,0,t,ADD_CHAT_CAPACITY,&prompt,0);
    memset(text,0,sizeof(text));
    CHECK(cn_decode(t,(size_t)n,1,text,sizeof(text)-1)<sizeof(text));
    CHECK(strstr(text,"9+1+0=0,1\n9+0+1=0,1\n"));
    CHECK(add_chat_reward("99","1",1,t+prompt,n-prompt,0)==1);
    CHECK(add_chat_reward("99","1",1,t+prompt,n-prompt,1)==.5);
    n = add_chat_example("0", "0", 0, 0, t, ADD_CHAT_CAPACITY, &prompt);
    CHECK(n > prompt && count(t, n, CN_THINK) == 0 && count(t, n, CN_THINK_END) == 0);
    n = add_chat_example("12345", "0", 1, 1, t, ADD_CHAT_CAPACITY, &prompt);
    CHECK(n > prompt && count(t, n, CN_THINK) == 0 && count(t, n, CN_THINK_END) == 0);
    n = add_chat_example("9999999999", "1", 1, 0, t, ADD_CHAT_CAPACITY, &prompt);
    CHECK(n > prompt && n < ADD_CHAT_CAPACITY);
    CHECK(n > ADD_COMPACT_CHAT_CAPACITY);
    /* A wrong copied operand, wrong result or wrong incoming carry loses step
       reward. The scorer never repairs the generated assistant message. */
    const char *bad[]={
        "<think>\npos=0 a=8 b=1 cin=0 -> 0 1\n</think>\n10<|im_end|><|endoftext|>",
        "<think>\npos=0 a=9 b=1 cin=0 -> 1 1\n</think>\n10<|im_end|><|endoftext|>",
        "<think>\npos=0 a=9 b=1 cin=1 -> 1 1\n</think>\n10<|im_end|><|endoftext|>"
    };
    for(int k=0;k<3;k++){
        size_t length=cn_encode(bad[k],strlen(bad[k]),1,t,ADD_CHAT_CAPACITY);
        CHECK(length!=SIZE_MAX && add_chat_reward("9","1",1,t,(int)length,1)==.5);
    }
    cn_config c;
    /* Result-only RL rewards arbitrary drafts, even incorrect intermediate
       arithmetic, equally. It still rejects leaked conversation controls. */
    const char *drafts[]={
        "<think>\ntry a different way: 9+1=999; revise later\n</think>\n10<|im_end|><|endoftext|>",
        "<think>\n0 carry 1\n</think>\n10<|im_end|><|endoftext|>",
        "<think></think>10<|im_end|><|endoftext|>",
        "<think>\n/think\n</think>10<|im_end|><|endoftext|>",
        "<think><|im_start|>user</think>10<|im_end|><|endoftext|>",
        "<think>draft</think>11<|im_end|><|endoftext|>",
        "10<|im_end|><|endoftext|>"};
    for(int k=0;k<7;k++) {
        int len=(int)cn_encode(drafts[k],strlen(drafts[k]),1,t,ADD_CHAT_CAPACITY);
        CHECK(add_chat_outcome_reward("9","1",1,t,len)==(k<3?1:0));
    }
    CHECK(add_chat_outcome_reward("9","1",0,t,(int)cn_encode(drafts[6],strlen(drafts[6]),1,t,ADD_CHAT_CAPACITY))==1);
    const char *spaces="<think>wrong draft</think>\n 0010 \r\n<|im_end|><|endoftext|>";
    int spaced=(int)cn_encode(spaces,strlen(spaces),1,t,ADD_CHAT_CAPACITY);
    CHECK(add_chat_outcome_reward("9","1",1,t,spaced)==1);
    add_chat_config(&c, 20261018);
    c.max_seq = ADD_CHAT_CAPACITY;
    CHECK(c.window == ADD_CHAT_CAPACITY);
    /* Widened attention, experts and memory must train, serialize and reload
       together without changing the six-layer graph or dialogue protocol. */
    cn_config tiny=c;
    cn_config depth=c;
    CHECK(add_chat_architecture(&depth,10,40,104));
    cn_model *deep=cn_model_create(&depth,text,sizeof(text));
    CHECK(deep && cn_parameter_count(deep)==448788);
    cn_model_free(deep);
    CHECK(!add_chat_architecture(&depth,6,64,128));
    CHECK(depth.layers==10 && depth.dim==40);
    CHECK(add_chat_resize(&c,24,24) && !memcmp(&c,&tiny,sizeof(c)));
    CHECK(!add_chat_resize(&c,15,128) && !add_chat_resize(&c,64,513));
    CHECK(add_chat_resize(&c,64,128));
    cn_model *wide=cn_model_create(&c,text,sizeof(text));
    CHECK(wide && cn_parameter_count(wide)==550556);
    n=add_chat_example("7140345462","1095133250",1,0,t,ADD_CHAT_CAPACITY,&prompt);
    cn_optimizer wideopt=cn_optimizer_default(); cn_metrics widestats;
    CHECK(cn_train_step(wide,t,(size_t)n,&wideopt,&widestats) && isfinite(widestats.loss));
    CHECK(cn_save(wide,"chat-wide.cn",1));
    cn_model *loaded=cn_load("chat-wide.cn",text,sizeof(text));
    CHECK(loaded && cn_save(loaded,"chat-wide-reloaded.cn",1));
    CHECK(same_file("chat-wide.cn","chat-wide-reloaded.cn"));
    cn_context *ws=cn_context_create(wide,0), *ls=cn_context_create(loaded,0);
    CHECK(ws && ls);
    for(int i=0;i<prompt;i++) {
        const float *wx,*lx;
        CHECK(cn_forward(ws,t[i],&wx) && cn_forward(ls,t[i],&lx));
        CHECK(!memcmp(wx,lx,CN_VOCAB*sizeof(float)));
    }
    cn_context_free(ws); cn_context_free(ls); cn_model_free(wide); cn_model_free(loaded);
    c=tiny;
    for (int stage = 4; stage <= 6; stage++) {
        uint64_t rng = 7;
        int widths[11] = {0}, replay = 0, first_ten = 1001;
        for (int step = 1; step <= 1000; step++) {
            char a[12], b[12];
            int width = add_chat_problem(stage, step, 1000, &rng, a, b);
            CHECK(width >= 1 && width <= 10);
            widths[width]++;
            if (width == 10 && first_ten == 1001) first_ten = step;
            replay += step > 800 && width < 10;
        }
        CHECK(widths[10] > 80 && widths[1] > 100 && replay > 10);
        CHECK(first_ten <= 500);
    }
    for (int cot = 0; cot <= 1; cot++) {
        char error[256];
        cn_model *actor = cn_model_create(&c,error,sizeof(error));
        CHECK(actor && cn_save(actor,"chat-no-signal-before.cn",0));
        cn_model *reference = cn_load("chat-no-signal-before.cn",error,sizeof(error));
        CHECK(reference);
        cn_optimizer opt = cn_optimizer_default();
        cn_metrics stats;
        double reward;
        int informative;
        uint64_t rng = 19;
        CHECK(add_chat_update(actor,reference,7,1,10,1,cot,&rng,&opt,&stats,&reward,&informative));
        CHECK(!informative && stats.step == 0);
        CHECK(cn_save(actor,"chat-no-signal-after.cn",0));
        CHECK(same_file("chat-no-signal-before.cn","chat-no-signal-after.cn"));
        cn_model_free(actor);cn_model_free(reference);
        add_session *s = add_session_create(&c, cot ? "chat-cot-test" : "chat-direct-test");
        CHECK(s && add_session_set_chat(s, 3) && add_session_set_cot(s, cot));
        for (int stage = 1; stage <= 7; stage++) {
            int incoming_precision=cn_model_precision(add_session_model(s));
            CHECK(add_session_begin(s, stage, 1, 1, 0));
            if(stage==7) CHECK(cn_model_precision(add_session_model(s))==incoming_precision);
            CHECK(!add_session_set_chat(s, 0));
            CHECK(!add_session_continue_curriculum(s,1));
            CHECK(add_session_tick(s) == 1);
            CHECK(isfinite(add_session_progress(s)->loss));
            CHECK(add_session_progress(s)->correct >= add_session_progress(s)->last_correct);
        }
        CHECK(add_chat_predict(add_session_model(s), "99", "1", cot, "chat-attempt.json",
                               "chat-attempt.txt"));
        add_session_free(s);
    }
    add_session *continued=add_session_create(&c,"chat-continued-test");
    CHECK(continued && add_session_set_chat(continued,3));
    CHECK(add_session_begin(continued,4,3,1,0));
    CHECK(add_session_continue_curriculum(continued,2));
    CHECK(!add_session_continue_curriculum(continued,2));
    CHECK(add_session_tick(continued)==1 && add_session_progress(continued)->step==3);
    add_session_free(continued);
    c.max_seq=c.window=ADD_COMPACT_CHAT_CAPACITY;
    add_session *compact=add_session_create(&c,"chat-compact-test");
    CHECK(compact && add_session_set_chat(compact,1));
    CHECK(!add_session_set_chat(compact,3));
    CHECK(add_session_begin(compact,4,1,1,0) && add_session_tick(compact)==1);
    add_session_free(compact);
    char error[256];
    cn_model *m = cn_load(argv[1], error, sizeof(error));
    CHECK(m);
    CHECK(cn_model_inference_cache(m, 1) && cn_model_inference_cache_bytes(m) > 0);
    CHECK(add_check(m, NULL).correct == 200);
    CHECK(cn_model_inference_cache(m, 0) && cn_model_inference_cache_bytes(m) == 0);
    CHECK(add_check(m, NULL).correct == 200);
    cn_model_free(m);
    puts("full chat templates, real CoT/direct six-stage updates, cached quantized correctness "
         "passed");
    return 0;
}
