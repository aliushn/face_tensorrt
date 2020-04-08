#include "mtcnn.h"

MtcnnDetector::MtcnnDetector()
{
}

MtcnnDetector::~MtcnnDetector()
{
    this->Pnet.Clear();
    this->Rnet.Clear();
    this->Onet.Clear();
    this->Lnet.Clear();
}

void MtcnnDetector::Init(string model_foler){
    vecto<string> onnx_files = {
            model_foler + "res/onnx/det1.onnx"
    };

    vecto<string> onnx_files = {
            model_foler + "res/onnx/det2.onnx"
    };

    vecto<string> onnx_files = {
            model_foler + "res/onnx/det3.onnx"
    };

    vecto<string> caffe_model_files = {
            model_foler + "res/caffemodel/det4.caffemodel"
    };

    vecto<string> caffe_protxt_files = {
            model_foler + "res/caffemodel/det4.prototxt"
    };

    this->Pnet.load_param(param_files[0].c_str());
    this->Pnet.load_param(param_files[0].c_str());
    this->Pnet.load_param(param_files[0].c_str());
    this->Pnet.load_param(param_files[0].c_str());
    this->Pnet.load_param(param_files[0].c_str());

    printf()
}

