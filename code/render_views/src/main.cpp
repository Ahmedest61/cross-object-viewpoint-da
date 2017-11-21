#include <igl/viewer/Viewer.h>
#include <igl/readObj.h>
#include <iostream>
#include <igl/png/writePNG.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <stdlib.h>

#define NUM_RENDS 150
#define BACKGROUND false 
#define PI 3.14159265358979

// Input args
std::vector<std::string> *mesh_filepaths;
std::string output_dir;
int width;
int height;
std::string version;

void captureImages(igl::viewer::Viewer& viewer) {

  // Initialize png buffers
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(width, height);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(width, height);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(width, height);
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(width, height);

  // Initialize vars needed for camera rotations
  Eigen::Matrix3f xRotate;
  Eigen::Matrix3f yRotate;
  Eigen::Matrix3f zRotate;
  std::cout << "Rendering images for " << mesh_filepaths->size() 
    << " models" << std::endl;

  // Create output directory
  std::stringstream new_dir;
  new_dir << output_dir << "/" << "V" << version;
  std::stringstream new_dir_cmd;
  new_dir_cmd << "mkdir " << new_dir.str();
  const char* cmd = (new_dir_cmd.str()).c_str();
  system(cmd);

  // Create output annotation file
  std::stringstream annot_name;
  annot_name << new_dir.str() << "/" << "annots" << ".txt";
  std::ofstream annotFile;
  annotFile.open(annot_name.str());

  int model_ctr = 0;
  for (int i = 0; i < mesh_filepaths->size(); i++) {

    // Load mesh
    std::cout << "Model: " << i+1 << "/" << mesh_filepaths->size() << std::endl;
    std::string mesh_fp = mesh_filepaths->at(i);
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(mesh_fp, V, F);

    // Orient so image facing forwards
    double angle_x = 1.57;
    double angle_y = 0;
    double angle_z = -1.57;
    Eigen::MatrixXd rotation_x(3,3);
    Eigen::MatrixXd rotation_y(3,3);
    Eigen::MatrixXd rotation_z(3,3);
    rotation_x << 
      1, 0, 0,
      0, std::cos(angle_x), std::sin(angle_x),
      0, 0-std::sin(angle_x), std::cos(angle_x);
    rotation_y << 
      std::cos(angle_y), 0, 0-std::sin(angle_y),
      0, 1, 0,
      std::sin(angle_y), 0, std::cos(angle_y);
    rotation_z <<
      cos(angle_z), sin(angle_z), 0,
      -sin(angle_z), cos(angle_z), 0,
      0, 0, 1;
    V = V * rotation_x;
    V = V * rotation_y;
    V = V * rotation_z;

    // Draw mesh to viewer
    viewer.data.clear();
    viewer.data.set_mesh(V,F);
    viewer.core.align_camera_center(viewer.data.V,viewer.data.F);
    viewer.draw();

    // Perform NUM_RENDS random renderings
    int im_ctr = 0;
    for (int j = 0; j < NUM_RENDS; j++) {

      // Generate rendering parameters
      float elevation = rand() % 135;
      float azimuth = rand() % 360;
      float elevation_rad = elevation * PI/180;
      float azimuth_rad = azimuth * PI/180;

      // Calc rotations
      xRotate << 
        1, 0, 0,
        0, std::cos(elevation_rad), std::sin(elevation_rad),
        0, 0-std::sin(elevation_rad), std::cos(elevation_rad);
      zRotate <<
        std::cos(azimuth_rad), std::sin(azimuth_rad), 0,
        -std::sin(azimuth_rad), std::cos(azimuth_rad), 0,
        0, 0, 1;

      // TODO: vary zoom

      // Rotate mesh and render
      Eigen::Quaternionf rot(xRotate*zRotate);
      viewer.core.trackball_angle = rot;
      viewer.draw();

      // Draw view to RGBA buffers
      viewer.core.draw_buffer(viewer.data, viewer.opengl, false, R,G,B,A);

      // If desired, include background in screenshot
      if (BACKGROUND) {
        for (int y = 0; y < A.rows(); y++) {
          for (int z = 0; z < A.cols(); z++) {
            A(y,z) = char(255);
          }
        }
      }

      // Take PNG screenshot and save to file
      std::stringstream out_name;
      out_name << new_dir.str() << "/" << model_ctr << "_" << im_ctr << ".png";
      igl::png::writePNG(R,G,B,A,out_name.str());

      // Write annotation
      float outElevation = -1 * (elevation - 135) - 45;
      float outAzimuth = azimuth;
      annotFile << model_ctr << "_" << im_ctr << "," << outElevation << "," << outAzimuth << std::endl;
      im_ctr++;

    }

    // Reset for next model
    viewer.data.V = Eigen::MatrixXd();
    viewer.data.F = Eigen::MatrixXi();
    model_ctr++;
  }
  std::cout << "--RENDERING FINISHED--" << std::endl;
  annotFile.close();
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
  if (argc < 6) {
    std::cout << "~~ERROR~~" << std::endl;
    std::cout << 
      "Usage: ./bin FILEPATH_OF_LIST_OF_FILES.txt OUTPUT_DIRECTORY WIDTH HEIGHT VERSION_NAME"
      << std::endl;
    return -1;
  }
  std::string mesh_listing_file = argv[1];
  output_dir = argv[2];
  mesh_filepaths = readLines(mesh_listing_file);
  width = std::atoi(argv[3]);
  height = std::atoi(argv[4]);
  version = argv[5];

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
