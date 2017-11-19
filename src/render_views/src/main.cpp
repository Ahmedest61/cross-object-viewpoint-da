#include <igl/viewer/Viewer.h>
#include <igl/readObj.h>
#include <iostream>
#include <igl/png/writePNG.h>
#include <unistd.h>
#include <vector>
#include <fstream>

#define PI 3.14159265358979

std::vector<std::string> *mesh_filepaths;
std::string output_dir;
int width;
int height;

void captureImages(igl::viewer::Viewer& viewer) {

  // Initialize png buffers
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

  // Initialize vars needed for camera rotations
  bool background = true;
  int x_jumps = 7;
  float x_angles[x_jumps];
  x_angles[0] = PI/2;
  x_angles[1] = PI/3;
  x_angles[2] = PI/6;
  x_angles[3] = 0;
  x_angles[4] = -PI/6;
  x_angles[5] = -PI/3;
  x_angles[6] = -PI/2;
  int y_jumps = 30;
  Eigen::Matrix3f xRotate;
  Eigen::Matrix3f yRotate;
  std::cout << "Rendering images for " << mesh_filepaths->size() 
    << " models" << std::endl;
  
  for (int i = 0; i < mesh_filepaths->size(); i++) {

    std::cout << i+1 << std::endl;
    // Load mesh
    std::string mesh_fp = mesh_filepaths->at(i);
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(mesh_fp, V, F);

    // Create output directory
    std::stringstream new_dir;
    std::string filename = mesh_fp.substr(0, mesh_fp.size()-4);
    int begin = filename.rfind('/');
    filename = filename.substr(begin + 1);
    new_dir << output_dir << "/" << filename;
    std::stringstream new_dir_cmd;
    new_dir_cmd << "mkdir " << new_dir.str();
    const char* cmd = (new_dir_cmd.str()).c_str();
    system(cmd);

    // Orient so image facing forwards
    //   TODO: hardcoded. need to read from metadata file
    double angle_x = -1.57;
    double angle_y = 1.57;
    Eigen::MatrixXd rotation_x(3,3);
    Eigen::MatrixXd rotation_y(3,3);
    rotation_x << 
      1, 0, 0,
      0, std::cos(angle_x), std::sin(angle_x),
      0, 0-std::sin(angle_x), std::cos(angle_x);
    rotation_y << 
      std::cos(angle_y), 0, 0-std::sin(angle_y),
      0, 1, 0,
      std::sin(angle_y), 0, std::cos(angle_y);
    V = V * rotation_x;
    V = V * rotation_y;

    // Draw mesh to viewer
    viewer.data.clear();
    viewer.data.set_mesh(V,F);
    viewer.core.align_camera_center(viewer.data.V,viewer.data.F);
    viewer.draw();

    // Render in multiple views
    int im_ctr = 0;
    for (int j = 0; j < x_jumps; j++) {
      xRotate << 
        1, 0, 0,
        0, std::cos(x_angles[j]), std::sin(x_angles[j]),
        0, 0-std::sin(x_angles[j]), std::cos(x_angles[j]);

      for (int k = 0; k < y_jumps; k++) {
        float y_angle = ((2*PI)/float(y_jumps))*float(k);
        yRotate <<
          std::cos(y_angle), 0, -std::sin(y_angle),
          0, 1, 0,
          std::sin(y_angle), 0, std::cos(y_angle);

        // Rotate mesh and render
        Eigen::Quaternionf rot(xRotate*yRotate);
        viewer.core.trackball_angle = rot;
        viewer.draw();

        // Draw view to RGBA buffers
        viewer.core.draw_buffer(viewer.data, viewer.opengl, false, R,G,B,A);

        // If desired, include background in screenshot
        if (background) {
          for (int y = 0; y < A.rows(); y++) {
            for (int z = 0; z < A.cols(); z++) {
              A(y,z) = char(255);
            }
          }
        }

        // Take PNG screenshot and save to file
        std::stringstream out_name;
        out_name << new_dir.str() << "/" << im_ctr << ".png";
        igl::png::writePNG(R,G,B,A,out_name.str());
        im_ctr++;
      }
    }
    viewer.data.V = Eigen::MatrixXd();
    viewer.data.F = Eigen::MatrixXi();
  }
  std::cout << "--RENDERING FINISHED--" << std::endl;
}

bool key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifier) {
  if (key == ' ') {
    captureImages(viewer);
  }   
  return false;
}

std::vector<std::string>* readLines(std::string filepath) {

  std::vector<std::string> *lines = new std::vector<std::string>();
  std::ifstream file(filepath);
  std::string line;
  if (file.is_open()) {
    while (getline(file, line)) {
      lines->push_back(line);
    }
    file.close();
  }
  return lines;
}

int main(int argc, char *argv[])
{

  // Get command line args
  if (argc < 5) {
    std::cout << "~~ERROR~~" << std::endl;
    std::cout << 
      "Usage: ./bin FILEPATH_OF_LIST_OF_FILES.txt OUTPUT_DIRECTORY WIDTH HEIGHT" 
      << std::endl;
    return -1;
  }
  std::string mesh_listing_file = argv[1];
  output_dir = argv[2];
  mesh_filepaths = readLines(mesh_listing_file);
  width = std::atoi(argv[3]);
  height = std::atoi(argv[4]);

  // Bring up viewer and prep for capture
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  std::cout << "Press space bar to begin capturing" << std::endl;
  igl::viewer::Viewer viewer;
  viewer.callback_key_down = &key_down;
  viewer.data.set_mesh(V, F);
  viewer.data.set_face_based(false);
  viewer.data.set_normals(viewer.data.V_normals);
  viewer.launch();
}
