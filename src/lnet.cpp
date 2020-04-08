//
// Created by dylee on 2020-03-22.
//

//#include "blackListDetection.h"

//BlackListDetection bld;

#include "lnet.h"


bool lnet::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

//    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());


    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

//    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
//    assert(mInputDims.nbDims == 4);

//    assert(network->getNbOutputs() == 1);
    mOutput1Dims = network->getOutput(0)->getDimensions();
    mOutput2Dims = network->getOutput(1)->getDimensions();
//    mOutput3Dims = network->getOutput(2)->getDimensions();
    int NbOutputsPnet = network->getNbOutputs();




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
bool lnet::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
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
    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
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
bool lnet::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
//    assert(mParams.inputTensorNames.size() == 1);

    if (!processInput(buffers))
    {
        return false;
    }
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool lnet::processInput(const samplesCommon::BufferManager& buffers)
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

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.lNetinputTensorNames[0]));
    for (int i = 0; i < inputH * inputW * inputC; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}





// Make face features of BlackList
//void checkBlacklist::makeBlackListFeatures(string path)
//{
//    vector<string> blacklistFiles = getImageFileListFromFolder(path);
//
//    for (int i=0; i < blacklistFiles.size();i++){
//        string full_path = blacklistFiles.at(i);
//        std::cout<<"blacklist path is : " <<full_path.c_str()<<std::endl;
//
//        assert(full_path.c_str());
//        cv::Mat input = cv::imread(full_path.c_str(),1);
//    }
//}



//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool lnet::verifyOutput(const samplesCommon::BufferManager& buffers)
{



//    const int output1Size = mOutput1Dims.d[0];
//    const int output2Size = mOutput2Dims.d[1];
//    const int output3Size = mOutput3Dims.d[2];
    const int output1Size = mOutput1Dims.nbDims;
    const int output2Size = mOutput2Dims.nbDims;

//    std::cout<< "outputSize : " << outputSize <<std::endl;

    gLogInfo <<" output1Size " <<  output1Size << std::endl;
    gLogInfo <<" output2Size " <<  output2Size << std::endl;

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


lnet::~lnet() {
//    m_LandmarkNet.clear();
//    m_ConditionCheckNet.clear();
//    m_CardCheckNet.clear();
}

int lnet::Init(int imageFormat, int imageWidth, int imageHeight, int rotateDegree, float imageMargin, float stableRatio1, int stableCount, std::string modelsPath)
{
//#ifdef PRINT_LOG
//    LOGD("checkBlackList Initialization Start")
//#endif
//    std::string
}







