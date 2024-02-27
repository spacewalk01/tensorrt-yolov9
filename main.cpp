#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "yolov9.h"


bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

int main(int argc, char** argv)
{
    const string engine_file_path{ argv[1] };
    const string path{ argv[2] };
    vector<string> imagePathList;
    bool                     isVideo{ false };
    assert(argc == 3);

    if (IsFile(path)) 
    {
        string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        glob(path + "/*.jpg", imagePathList);
    }

    // Assume it's a folder, add logic to handle folders
    // init model
    Yolov9 model(engine_file_path);

    if (isVideo) {
        //path to video
        string VideoPath = path;
        // open cap
        VideoCapture cap(VideoPath);

        int width = cap.get(CAP_PROP_FRAME_WIDTH);
        int height = cap.get(CAP_PROP_FRAME_HEIGHT);

        // Create a VideoWriter object to save the processed video
        VideoWriter output_video("output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(width, height));
        while (1)
        {
            Mat frame;
            cap >> frame;

            if (frame.empty()) break;

            auto start = chrono::system_clock::now();
            
            vector<Detection> bboxes;
            model.predict(frame, bboxes);

            auto end = chrono::system_clock::now();
            cout << "Time of per frame: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

            model.draw(frame, bboxes);

            imshow("prediction", frame);
            output_video.write(frame);
            waitKey(1);
        }

        // Release resources
        destroyAllWindows();
        cap.release();
        output_video.release();
    }
    else {
        // path to folder saves images
        string imageFolderPath_out = "results/";
        for (const auto& imagePath : imagePathList)
        {
            // open image
            Mat frame = imread(imagePath);
            if (frame.empty())
            {
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }

            auto start = chrono::system_clock::now();

            vector<Detection> bboxes;
            model.predict(frame, bboxes);
            model.draw(frame, bboxes);

            auto end = chrono::system_clock::now();
            cout << "Time of per frame: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
            
            istringstream iss(imagePath);
            string token;
            while (getline(iss, token, '/'))
            {
            }
            imwrite(imageFolderPath_out + token, frame);
            cout << imageFolderPath_out + token << endl;

            imshow("prediction", frame);
            waitKey(0);
        }
    }
    
    return 0;
}
