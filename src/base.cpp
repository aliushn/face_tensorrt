//
// Created by dylee on 2020-03-21.
//

#include "base.h"

//void getAffineMatrix(float* src_5pts, const float* dst_5pts, float* M){
//    float src[10], dst[10];
//    memcpy(src,src_5pts,sizeof(float) * 10);
//    memcpy(dst,dst_5pts,sizeof(float) * 10);
//
//    float ptmp[2];
//    ptmp[0] = ptmp[1] = 0;
//    for (int i=0;i<5;++i){
//        ptmp[0] += src[i];
//        ptmp[1] += src[5+i];
//    }
//    ptmp[0]/=5;
//    ptmp[1]/=5;
//    for (int i=0;i<5;++i){
//        src[i] -= ptmp[0];
//        src[5+i]-= ptmp[1];
//        dst[i] -= ptmp[0];
//        dst[5+i] -= ptmp[1];
//    }
//
//    float dst_x = (dst[3] + dst[4] - dst[0] - dst[1]) / 2, dst_y = (dst[8] + dst[9] - dst[5] - dst[6]) / 2;
//    float src_x = (src[3] + src[4] - src[0] - src[1]) / 2, src_y = (src[8] + src[9] - src[5] - src[6]) / 2;
//    float theta = atan2(dst_x,dst_y) - atan2(src_x, src_y);
//
//    float scale = sqrt(pow(dst_x,2) +pow(dst_y,2)) / sqrt(pow(src_x,2) +pow(src_y,2));
//
//    float pts1[10];
//    float pts0[2];
//
//    float _a = sin(theta), _b=cos(theta);
//    pts0[0]=pts0[1]=0;
//    for (int i=0;i<5;++i){
//        pts1[i] = scale*(src[i]*_b + src[i+5]*_a);
//        pts1[i+5] = scale * (-src[i]*_a+src[i+5]*_b);
//        pts0[0]+=(dst[i] - pts1[i]);
//        pts0[1]+=(dst[i+5]-pts1[i+5]);
//    }
//
//    pts0[0] /= 5;
//    pts0[1] /= 5;
//
//    float sqloss = 0;
//    for (int i=0; i<5; ++i){
//        sqloss+= ((pts0[0]+pts1[i]-dst[i])*(pts0[0]+pts1[i]-dst[i])
//                +(pts0[1]+pts1[i+5]-dst[i+5])*(pts0[1]+pts1[i+5]-dst[i+5]));
//    }
//    float square_sum = 0;
//    for (int i=0;i<10;++i){
//        square_sum+=src[i]*src[i];
//    }
//    for (int t =0; t<200;++t){
//        _a = 0;
//        _b = 0;
//        for (int i=0;i<5;++i){
//            _a+=((pts0[0]-dst[i])*src[i+5]-(pts))
//        }
//    }
//}



