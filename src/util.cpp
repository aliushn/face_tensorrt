//
// Created by dylee on 2020-03-19.
//

#include "util.h"

std::vector<std::string> getImageFileListFromFolder(std::string folder)
{
    std::string path = folder + "*.jpg";
    std::vector<cv::String> filenames;
    std::vector<std::string> result;

    cv::glob(path.c_str(),filenames,true);

    for (int idx=0; idx<filenames.size();idx++)
    {
        result.push_back(filenames.at(idx).operator std::string());
        std::cout << result.at(idx) << std::endl;
    }
    return result;
}

std::vector<std::string> GetFeautreFileListFromFolder(std::string folder)
{
    std::string path = folder + "*.csv";
    std::vector<cv::String> filenames;
    std::vector<std::string> result;

    cv::glob(path.c_str(),filenames,true);

    for (int idx=0; idx<filenames.size();idx++) {
        result.push_back(filenames.at(idx).operator std::string());
        std::cout << result.at(idx) << std::endl;
    }
    return result;
}


// Change a file extension to a specific target
void ChangeExt(char *path, char *newext) {
#ifdef WIN32
    char drive[_MAX_DRIVE];
    char drive[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];

    _splitpath_s(path,drive,dir,fname,ext);
    sprinf(path, "%s%s%s",drive,dir,fname,newext);
#else
    char *p_result;
    char *p_tmp;
    char p_path;

    assert(path);

//    if ((p_tmp = strdup(path)) != NULL) {
//        std::experimental::filesystem::path p_path = path;
//        p_path.replace_extension(newext);
//        free(p_tmp);
//        std::cout << "p_path: " << p_path << std::endl;
////        sprintf(path, "%s", p_path);
//    }
#endif
}


std::vector<float> read_csv(int row, const char *filename) {
    std::vector<float> data;

    std::ifstream file(filename);
    while (file.good()) {
        std::vector<std::string> row = csv_read_row(file, ',');

        if (!row[0].find("#")) {
            continue;
        } else {
            for (int i = 0, leng = row.size(); i < leng; i++) {
                data.push_back(stof(row[i]));
            }
        }
    }
    file.close();
    return data;
}

std::vector<std::string> csv_read_row(std::istream &file, char delimiter)
{
    std::stringstream ss;
    bool inquotes = false;
    std::vector<std::string> row;

    while (file.good()){
        char c = file.get();
        if (!inquotes && c == '"'){
            inquotes = true;
        }
        else if (inquotes && c == '"'){
            if (file.peek() == '"'){
                ss << (char)file.get();
            }
            else{
                inquotes = false;
            }
        }
        else if (!inquotes && c == delimiter){
            row.push_back(ss.str());
            ss.str("");
        }
        else if (!inquotes && (c == '\r' || c == '\n')){
            if (file.peek() == '\n') { file.get(); }
            row.push_back(ss.str());
            return row;
        }
        else{
            ss << c;
        }
    }
}















