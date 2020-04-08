//
// Created by dylee on 2020-03-10.
//

//#include "blackListDetection.h"

//BlackListDetection bld;

#include "checkBlacklist.h"
#include <iostream>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
#include "logger.h"

bool checkBlacklist::build()
{
//    build MTCNN networks
    auto builder_pnet = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder_pnet)
    {
        return false;
    }

    auto builder_rnet = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder_rnet)
    {
        return false;
    }

    auto builder_onet = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder_onet)
    {
        return false;
    }

    auto builder_lnet = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder_lnet)
    {
        return false;
    }

//    define MTCNN networks
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto pnet_network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder_pnet->createNetworkV2(explicitBatch));
    if (!pnet_network)
    {
        return false;
    }

    auto rnet_network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder_rnet->createNetworkV2(explicitBatch));
    if (!rnet_network)
    {
        return false;
    }

    auto onet_network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder_onet->createNetworkV2(explicitBatch));
    if (!onet_network)
    {
        return false;
    }

    auto lnet_network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder_lnet->createNetworkV2(explicitBatch));
    if (!lnet_network)
    {
        return false;
    }

//    define MTCNN config
    auto pnet_config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder_pnet->createBuilderConfig());
    if (!pnet_config)
    {
        return false;
    }
    auto rnet_config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder_rnet->createBuilderConfig());
    if (!rnet_config)
    {
        return false;
    }
    auto onet_config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder_onet->createBuilderConfig());
    if (!onet_config)
    {
        return false;
    }auto lnet_config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder_lnet->createBuilderConfig());
    if (!lnet_config)
    {
        return false;
    }

//    define MTCNN onnx parser
    auto pnet_parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*pnet_network, gLogger.getTRTLogger()));
    if (!pnet_parser)
    {
        return false;
    }

    auto rnet_parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*rnet_network, gLogger.getTRTLogger()));
    if (!rnet_parser)
    {
        return false;
    }

    auto onet_parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*onet_network, gLogger.getTRTLogger()));
    if (!onet_parser)
    {
        return false;
    }
//    define MTCNN lnet caffe parser
    auto lnet_parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    if (!lnet_parser)
    {
        return false;
    }

//    construct MTCNN
    auto pnet_constructed = constructOnnxNetwork(builder_pnet, pnet_network, pnet_config, pnet_parser, 0);
    if (!pnet_constructed)
    {
        return false;
    }
    auto rnet_constructed = constructOnnxNetwork(builder_rnet, rnet_network, rnet_config, rnet_parser, 1);
    if (!rnet_constructed)
    {
        return false;
    }
    auto onet_constructed = constructOnnxNetwork(builder_onet, onet_network, onet_config, onet_parser, 2);
    if (!onet_constructed)
    {
        return false;
    }
    auto lnet_constructed = constructCaffeNetwork(builder_lnet, lnet_network, lnet_config, lnet_parser);
    if (!lnet_constructed)
    {
        return false;
    }


//    Generate MTCNN cudaEngine
    pnet_mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder_pnet->buildEngineWithConfig(*pnet_network, *pnet_config), samplesCommon::InferDeleter());
    if (!pnet_mEngine)
    {
        return false;
    }
    rnet_mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder_rnet->buildEngineWithConfig(*rnet_network, *rnet_config), samplesCommon::InferDeleter());
    if (!rnet_mEngine)
    {
        return false;
    }
    onet_mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder_onet->buildEngineWithConfig(*onet_network, *onet_config), samplesCommon::InferDeleter());
    if (!onet_mEngine)
    {
        return false;
    }
    lnet_mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder_lnet->buildEngineWithConfig(*lnet_network, *lnet_config), samplesCommon::InferDeleter());
    if (!lnet_mEngine)
    {
        return false;
    }

//    assert(network->getNbInputs() == 1);

    pnet_mInputDims = pnet_network->getInput(0)->getDimensions();
    pnet_mOutput1_Dims = pnet_network->getOutput(0)->getDimensions();
    pnet_mOutput2_Dims = pnet_network->getOutput(1)->getDimensions();

    rnet_mInputDims = rnet_network->getInput(0)->getDimensions();
    rnet_mOutput1_Dims = rnet_network->getOutput(0)->getDimensions();
    rnet_mOutput2_Dims = rnet_network->getOutput(1)->getDimensions();

    onet_mInputDims = onet_network->getInput(0)->getDimensions();
    onet_mOutput1_Dims = onet_network->getOutput(0)->getDimensions();
    onet_mOutput2_Dims = onet_network->getOutput(1)->getDimensions();
    onet_mOutput2_Dims = onet_network->getOutput(2)->getDimensions();

    pnet_mInputDims = pnet_network->getInput(0)->getDimensions();
    pnet_mOutput1_Dims = pnet_network->getOutput(0)->getDimensions();
    pnet_mOutput2_Dims = pnet_network->getOutput(1)->getDimensions();
    pnet_mOutput2_Dims = pnet_network->getOutput(1)->getDimensions();
    pnet_mOutput2_Dims = pnet_network->getOutput(1)->getDimensions();
    pnet_mOutput2_Dims = pnet_network->getOutput(1)->getDimensions();

//    assert(mOutputDims.nbDims == 2);
    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool checkBlacklist::constructOnnxNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                       SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                       SampleUniquePtr<nvonnxparser::IParser>& parser, int netIdx)
{
    string &netName = mParams.pNetOnnxFileName;
//    const std::string* netName;
    if (netIdx == 1){
        netName = mParams.rNetOnnxFileName;
    }else if (netIdx == 2){
        netName = mParams.oNetOnnxFileName;
    }

    auto parsed = parser->parseFromFile(
            locateFile(netName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
//        std::cout<<"fail constructing networks"<<std::endl;
        return false;
    }
    std::cout<<"fail constructing networks"<<std::endl;


    builder->setMaxBatchSize(mParams.rNetbatchSize);
    config->setMaxWorkspaceSize(16_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
    std::cout<<"pass constructing networks"<<std::endl;
    return true;
}


bool checkBlacklist::constructCaffeNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                            SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                            SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser)
{

    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
            locateFile(mParams.lNetprototxtFileName, mParams.dataDirs).c_str(),locateFile(mParams.lNetweightsFileName, mParams.dataDirs).c_str(),
            *network, nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.lNetoutputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    // add mean subtraction to the beginning of the network
//    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
//    mMeanBlob = SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>(parser->parseBinaryProto(mParams.meanFileName.c_str()));
//    nvinfer1::Weights meanWeights{nvinfer1::DataType::kFLOAT, mMeanBlob->getData(), inputDims.d[1] * inputDims.d[2]};
    // For this sample, a large range based on the mean data is chosen and applied to the head of the network.
    // After the mean subtraction occurs, the range is expected to be between -127 and 127, so the rest of the network
    // is given a generic range.
    // The preferred method is use scales computed based on a representative data set
    // and apply each one individually based on the tensor. The range here is large enough for the
    // network, but is chosen for example purposes only.
//    float maxMean = samplesCommon::getMaxValue(static_cast<const float*>(meanWeights.values), samplesCommon::volume(inputDims));

//    auto mean = network->addConstant(nvinfer1::Dims3(1, inputDims.d[1], inputDims.d[2]), meanWeights);
//    mean->getOutput(0)->setDynamicRange(-maxMean, maxMean);
//    network->getInput(0)->setDynamicRange(-maxMean, maxMean);
//    auto meanSub = network->addElementWise(*network->getInput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
//    meanSub->getOutput(0)->setDynamicRange(-maxMean, maxMean);
//    network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
//    samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}


//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool checkBlacklist::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager pnet_buffers(pnet_mEngine, mParams.batchSize);
    samplesCommon::BufferManager rnet_buffers(rnet_mEngine, mParams.batchSize);
    samplesCommon::BufferManager onet_buffers(onet_mEngine, mParams.batchSize);
    samplesCommon::BufferManager lnet_buffers(lnet_mEngine, mParams.batchSize);

    auto pnet_context = SampleUniquePtr<nvinfer1::IExecutionContext>(pnet_mEngine->createExecutionContext());
    if (!pnet_context)
    {
        return false;
    }
    auto rnet_context = SampleUniquePtr<nvinfer1::IExecutionContext>(rnet_mEngine->createExecutionContext());
    if (!rnet_context)
    {
        return false;
    }
    auto onet_context = SampleUniquePtr<nvinfer1::IExecutionContext>(onet_mEngine->createExecutionContext());
    if (!onet_context)
    {
        return false;
    }
    auto lnet_context = SampleUniquePtr<nvinfer1::IExecutionContext>(lnet_mEngine->createExecutionContext());
    if (!lnet_context)
    {
        return false;
    }





    if (!pnet_processInput(pnet_buffers))
    {
        return false;
    }

    float* hostDataBuffer = static_cast<float*>(pnet_buffers.getHostBuffer(mParams.pNetinputTensorNames[0]));
    for (int i = 0; i < inputH * inputW * inputC; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }



    // Memcpy from host input buffers to device input buffers
    pnet_buffers.copyInputToDevice();
    bool status = pnet_context->executeV2(pnet_buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }


    // Memcpy from device output buffers to host output buffers
    pnet_buffers.copyOutputToHost();
    // Verify results
    if (!verifyOutput(pnet_buffers))
    {
        return false;
    }


    if (!rnet_processInput(rnet_buffers))
    {
        return false;
    }
    if (!onet_processInput(onet_buffers))
    {
        return false;
    }
    if (!lnet_processInput(lnet_buffers))
    {
        return false;
    }




    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool checkBlacklist::pnet_processInput(const samplesCommon::BufferManager& buffers)
{
//    const int inputH = mInputDims.d[2];//350
//    const int inputW = mInputDims.d[3];//400
    const int inputH = 12;//350
    const int inputW = 12;//400
    const int inputC = 3;//400


    std::cout<<"inputH"<<inputH<<std::endl;
    std::cout<<"inputW"<<inputW<<std::endl;


    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW * inputC);
    mNumber = rand() % 10;
//    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    std::string faceimgs = locateFile("ljb.png", mParams.dataDirs);
//    matrix.col(0).copyTo(vec);
    cv::Mat face_img= cv::imread(faceimgs,cv::IMREAD_COLOR);

    vector<FaceInfo> results;
    int face_img_w = face_img.size().width;
    int face_img_h = face_img.size().height;
    float minl = face_img_w < face_img_h ? face_img_w : face_img_h;
    double scale = 12.0 / this->minsize;
    minl *= scale;
    vector<double> scales;
    while(minl > 12)
    {
        scales.push_back(scale);
        minl *= this->factor;
        scale *= this->factor;
    }
    std::vector<std::vector<uint8_t>> scales_data;

    for (auto it = scales.begin(); it !=scales.end();it++)
    {
        scale=(double)(*it);
        int hs = (int)ceil(face_img_h * scale);
        int ws = (int)ceil(face_img_w * scale);
        std::cout<<"resized image"<<std::endl;
        cv::resize(face_img,face_img,cv::Size(ws,hs));
        cv::subtract(face_img,this->mean_vals,face_img);
        cv::divide(face_img, norm_vals, face_img);
        cv::Mat flat = face_img.reshape(1, face_img.total()*face_img.channels());


        fileData = face_img.isContinuous()? flat : flat.clone();
        scales_data.push_back(fileData);
    }

//    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.pNetinputTensorNames[0]));
//    for (int i = 0; i < inputH * inputW * inputC; i++)
//    {
//        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
//    }

//    pnet_buffers.copyInputToDevice();
//    bool status = pnet_context->executeV2(pnet_buffers.getDeviceBindings().data());
//    if (!status)
//    {
//        return false;
//    }


    return fileData;
}

bool checkBlacklist::rnet_processInput(const samplesCommon::BufferManager& buffers)
{
//    const int inputH = mInputDims.d[2];//350
//    const int inputW = mInputDims.d[3];//400
    const int inputH = 12;//350
    const int inputW = 12;//400
    const int inputC = 3;//400


    std::cout<<"inputH"<<inputH<<std::endl;
    std::cout<<"inputW"<<inputW<<std::endl;


    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW * inputC);
    mNumber = rand() % 10;
//    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    std::string faceimgs = locateFile("ljb.png", mParams.dataDirs);
//    matrix.col(0).copyTo(vec);
    cv::Mat face_img= cv::imread(faceimgs,cv::IMREAD_COLOR);
    cv::resize(face_img,face_img,cv::Size(12,12));

    std::cout<<"resized image"<<std::endl;

    cv::Mat flat = face_img.reshape(1, face_img.total()*face_img.channels());
    fileData = face_img.isContinuous()? flat : flat.clone();

//    for (int i = 0; i < inputH * inputW * inputC; i++)
//    {
//        fileData[i] = uint8_t(face_img.at<uint8_t>());
//    }
//    std::cout<<"face_img"<<face_img.dims<<std::endl;





//    vector<float> vec;
//    face_img.col(0).copyTo(fileData);


    // Print an ascii representation
//    gLogInfo << "Input:" << std::endl;
//    for (int i = 0; i < inputH * inputW; i++)
//    {
//        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
//    }
//    gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.pNetinputTensorNames[0]));
    for (int i = 0; i < inputH * inputW * inputC; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}


bool checkBlacklist::onet_processInput(const samplesCommon::BufferManager& buffers)
{
//    const int inputH = mInputDims.d[2];//350
//    const int inputW = mInputDims.d[3];//400
    const int inputH = 12;//350
    const int inputW = 12;//400
    const int inputC = 3;//400


    std::cout<<"inputH"<<inputH<<std::endl;
    std::cout<<"inputW"<<inputW<<std::endl;


    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW * inputC);
    mNumber = rand() % 10;
//    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    std::string faceimgs = locateFile("ljb.png", mParams.dataDirs);
//    matrix.col(0).copyTo(vec);
    cv::Mat face_img= cv::imread(faceimgs,cv::IMREAD_COLOR);
    cv::resize(face_img,face_img,cv::Size(12,12));

    std::cout<<"resized image"<<std::endl;

    cv::Mat flat = face_img.reshape(1, face_img.total()*face_img.channels());
    fileData = face_img.isContinuous()? flat : flat.clone();

//    for (int i = 0; i < inputH * inputW * inputC; i++)
//    {
//        fileData[i] = uint8_t(face_img.at<uint8_t>());
//    }
//    std::cout<<"face_img"<<face_img.dims<<std::endl;





//    vector<float> vec;
//    face_img.col(0).copyTo(fileData);


    // Print an ascii representation
//    gLogInfo << "Input:" << std::endl;
//    for (int i = 0; i < inputH * inputW; i++)
//    {
//        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
//    }
//    gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.pNetinputTensorNames[0]));
    for (int i = 0; i < inputH * inputW * inputC; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}


bool checkBlacklist::lnet_processInput(const samplesCommon::BufferManager& buffers)
{
//    const int inputH = mInputDims.d[2];//350
//    const int inputW = mInputDims.d[3];//400
    const int inputH = 12;//350
    const int inputW = 12;//400
    const int inputC = 3;//400


    std::cout<<"inputH"<<inputH<<std::endl;
    std::cout<<"inputW"<<inputW<<std::endl;


    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW * inputC);
    mNumber = rand() % 10;
//    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    std::string faceimgs = locateFile("ljb.png", mParams.dataDirs);
//    matrix.col(0).copyTo(vec);
    cv::Mat face_img= cv::imread(faceimgs,cv::IMREAD_COLOR);
    cv::resize(face_img,face_img,cv::Size(12,12));

    std::cout<<"resized image"<<std::endl;

    cv::Mat flat = face_img.reshape(1, face_img.total()*face_img.channels());
    fileData = face_img.isContinuous()? flat : flat.clone();

//    for (int i = 0; i < inputH * inputW * inputC; i++)
//    {
//        fileData[i] = uint8_t(face_img.at<uint8_t>());
//    }
//    std::cout<<"face_img"<<face_img.dims<<std::endl;





//    vector<float> vec;
//    face_img.col(0).copyTo(fileData);


    // Print an ascii representation
//    gLogInfo << "Input:" << std::endl;
//    for (int i = 0; i < inputH * inputW; i++)
//    {
//        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
//    }
//    gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.pNetinputTensorNames[0]));
    for (int i = 0; i < inputH * inputW * inputC; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}



// Make face features of BlackList
void checkBlacklist::makeBlackListFeatures(string path)
{
    vector<string> blacklistFiles = getImageFileListFromFolder(path);

    for (int i=0; i < blacklistFiles.size();i++){
        string full_path = blacklistFiles.at(i);
        std::cout<<"blacklist path is : " <<full_path.c_str()<<std::endl;

        assert(full_path.c_str());
        cv::Mat input = cv::imread(full_path.c_str(),1);
    }
}



//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool checkBlacklist::verifyOutput(const samplesCommon::BufferManager& buffers)
{



//    const int output1Size = mOutput1Dims.d[0];
//    const int output2Size = mOutput2Dims.d[1];
//    const int output3Size = mOutput3Dims.d[2];
    const int output1Size = pnet_mOutput1_Dims.nbDims;
    const int output2Size = pnet_mOutput2_Dims.nbDims;

//    std::cout<< "outputSize : " << outputSize <<std::endl;

    gLogInfo <<" pnet_mOutput1_Dims output1Size " <<  output1Size << std::endl;
    gLogInfo <<" pnet_mOutput1_Dims output2Size " <<  output2Size << std::endl;

    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.pNetoutputTensorNames[0]));
    float* output2 = static_cast<float*>(buffers.getHostBuffer(mParams.pNetoutputTensorNames[1]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < output2Size; i++)
    {
        output2[i] = exp(output2[i]);
        sum += output2[i];
    }

    gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < output1Size; i++)
    {
       gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << std::endl;
        gLogInfo << "inference is finished" << i << std::endl;
    }
    gLogInfo << std::endl;
    gLogInfo << "inference is finished" << std::endl;


    gLogInfo << "Output1:" << std::endl;
    for (int i = 0; i < output2Size; i++)
    {
        output2[i] /= sum;
        val = std::max(val, output2[i]);
        if (val == output2[i])
        {
            idx = i;
        }

        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output2[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output2[i] * 10 + 0.5f)), '*') << std::endl;
        gLogInfo << "inference is finished" << i << std::endl;
    }
    gLogInfo << std::endl;
    gLogInfo << "inference is finished" << std::endl;

//    return idx == mNumber && val > 0.9f;
    return true;
}


//checkBlacklist::checkBlacklist() {
//    m_ImageFormat = 0;
//    m_ImageWidth = 1280;
//    m_ImageHeight = 720;
//    m_StableRatio1 = 0.08;        // XY displacement threshold ratio
//    m_StableCount = 10;           // Stable frame count
//    m_StableRatio3 = 0.7;          // Usable detected IDcard size ratio
//    m_CardCheckerFlag = 0;
//    m_FaceRocation = 0;
//    m_LandmarkInputWidth = 64;
//    m_LandmarkInputHeight = 64;
//    m_PrevLocationY = int(m_ImageWidth * 0.5);
//    m_WarpSize = cv::Size(CARD_WIDTH * 2, CARD_HEIGHT * 2);
//    m_WarpCorners = new std::vector<cv::Point2f>[MAX_WARP_CORNER];
//    m_StabilizedCorner = new std::vector<cv::Point2f>[1];
//    m_StabilizedCorner[0] = std::vector<cv::Point2f>(4);
//    m_CardType = 0;
//    for (int i = 0; i < MAX_CHEKER_STATE; i++) { m_CheckerResultState[i] = 0.0; }
//    for (int i = 0; i < MAX_WARP_CORNER; i++) { m_WarpCorners[i] = std::vector<cv::Point2f>(LANDMARK_POINT); }
//
//}


checkBlacklist::~checkBlacklist() {
//    m_LandmarkNet.clear();
//    m_ConditionCheckNet.clear();
//    m_CardCheckNet.clear();
}

int checkBlacklist::Init(int imageFormat, int imageWidth, int imageHeight, int rotateDegree, float imageMargin, float stableRatio1, int stableCount, std::string modelsPath)
{
//#ifdef PRINT_LOG
//    LOGD("checkBlackList Initialization Start")
//#endif
//    std::string
}







