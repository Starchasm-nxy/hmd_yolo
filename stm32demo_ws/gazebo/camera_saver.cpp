/*
 * camera_saver.cpp — Gazebo WorldPlugin
 * Saves pan-tilt camera frames to /tmp/gazebo_frame.jpg
 * Compile: see Makefile or run build.sh
 */

#include <gazebo/gazebo.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace gazebo {

class CameraSaver : public WorldPlugin
{
private:
    transport::NodePtr node;
    transport::SubscriberPtr imageSub;
    int frameCount = 0;

    void OnImage(ConstImageStampedPtr &msg)
    {
        int w = msg->image().width();
        int h = msg->image().height();
        int step = msg->image().step();
        const std::string &data = msg->image().data();

        if (w <= 0 || h <= 0 || data.empty())
            return;

        cv::Mat bgr(h, w, CV_8UC3);
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                int idx = r * step + c * 3;
                bgr.at<cv::Vec3b>(r, c) = cv::Vec3b(
                    (uchar)data[idx + 2],
                    (uchar)data[idx + 1],
                    (uchar)data[idx + 0]
                );
            }
        }

        cv::imwrite("/tmp/gazebo_frame.jpg", bgr);
        frameCount++;
        if (frameCount % 30 == 0)
            gzmsg << "[CameraSaver] " << frameCount << " frames saved" << std::endl;
    }

public:
    CameraSaver() : WorldPlugin() {}
    virtual ~CameraSaver() {}

    void Load(physics::WorldPtr /*world*/, sdf::ElementPtr /*sdf*/) override
    {
        gzmsg << "[CameraSaver] WorldPlugin loaded" << std::endl;

        transport::init();
        node = transport::NodePtr(new transport::Node());
        node->Init();

        imageSub = node->Subscribe(
            "/gazebo/tracking_world/pan_tilt_camera/tilt_link/camera/image",
            &CameraSaver::OnImage, this);

        gzmsg << "[CameraSaver] Subscribed to camera topic" << std::endl;
    }
};

GZ_REGISTER_WORLD_PLUGIN(CameraSaver)

} // namespace gazebo
