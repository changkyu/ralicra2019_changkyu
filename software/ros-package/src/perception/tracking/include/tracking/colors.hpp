#ifndef COLORS__HPP__
#define COLORS__HPP__

cv::Scalar GetColor(int idx)
{
    static std::vector<cv::Vec3b> colors;

    for (size_t i = colors.size(); i < (idx+1); i++)
    {
        int b = cv::theRNG().uniform(0, 255);
        int g = cv::theRNG().uniform(0, 255);
        int r = cv::theRNG().uniform(0, 255);
        colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    return cv::Scalar(colors[idx][0],colors[idx][1],colors[idx][2]);
}

#endif