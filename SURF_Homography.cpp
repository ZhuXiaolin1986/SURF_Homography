/**
 * @file SURF_Homography
 * @brief SURF detector + descriptor + FLANN Matcher + FindHomography
 * @author A. Huaman
 */

#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include <time.h>

# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/calib3d/calib3d.hpp"
# include "opencv2/nonfree/features2d.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

//////////////////////////////////////////////////////////////////////

#define ZOOM_IN_SCALE   (4)
#define PICS_SHOW_SCALE (0.25)
#define LINE_WIDTH    (8)
void show(const string &name, const Mat &image) {

    if (image.cols > 400 || image.rows > 400) {
        float scale = PICS_SHOW_SCALE;
        Mat img;
        resize(image, img, Size(), scale, scale);
        imshow(name, img);
    } else {
        imshow(name, image);
    }
}

void ReadMe()
{ printf("Usage: \n"
        "\tManu => SURF_Homography -m [-d] --scene=sceneImg.jpg\n"
        "\tAuto => SURF_Homography [-d] --scene=sceneImg.jpg --template=objectImg.jpg \n");
}


void onMouseForAutomatically( int event, int x, int y, int flags, void* param);
void onMouseForManually( int event, int x, int y, int flags, void* param);
void findObjectInScene(Mat object, Mat scene);
void getTimeStamp(char* buf);

String g_FileSaved;

bool g_DebugEnable = false;
bool g_OpManuMode = false;
string g_SceneFileName;
string g_ObjectFileName;

//////////////////////////////////////////////////////////////////////

int g_LeftTop_x = 0;
int g_LeftTop_y = 0;
int g_RightBottom_x = 0;
int g_RightBottom_y = 0;
int g_LeftButtonIsDown = 0;
int g_InProgress = 0;

Mat img_object;
Mat img_scene;
Mat img_scene_copy;
Mat scene_splitted;

int minHessian = 400;

//////////////////////////////////////////////////////////////////////

#define QT_FONT_NORMAL 50

String windowTitle = "Perspective Transformation Demo";
String labels[4] = { "TL","TR","BR","BL" };
vector< Point2f> roi_corners;
vector< Point2f> dst_corners(4);
int roiIndex = 0;
bool dragging;
int selected_corner_index = 0;
bool validation_needed = true;

//////////////////////////////////////////////////////////////////////

const char* keys =
{
    "{m |manually|false|manually or using a template }"
    "{d |debug|false|generate the splitted image or not }"
    "{s |scene|sceneImg.jpg|image to process }"
    "{t |template|template.jpg|the object you wish to find in scene }"
};


int main( int argc, char** argv )
{
  CommandLineParser parser(argc, argv, keys);
  g_OpManuMode = parser.get<bool>("manually");
  g_DebugEnable = parser.get<bool>("debug");
  g_SceneFileName = parser.get<string>("scene");
  g_ObjectFileName = parser.get<string>("template");
  printf("=================\n");
  parser.printParams();
  printf("=================\n");

  img_scene = imread( g_SceneFileName, CV_LOAD_IMAGE_GRAYSCALE );
  img_object = imread( g_ObjectFileName, CV_LOAD_IMAGE_GRAYSCALE );

  if(g_OpManuMode)
    goto MAUALLY_PROCESS;

AUTOMATICALLY_PROCESS:
  if( img_object.empty() || img_scene.empty() )
  {
    ReadMe();
    exit(-1);
  }

  img_scene_copy = img_scene.clone();
  namedWindow("scene");
  show("scene", img_scene_copy);
  setMouseCallback("scene", onMouseForAutomatically, 0);

  for(;;)
  {
    char c = (char)waitKey( 10 );

    // quit
    if ((c == 'q') | (c == 'Q') | (c == 27))
    {
        break;
    }

    // reset
    if ((c == 'r') | (c == 'R'))
    {
      destroyAllWindows();
      namedWindow("scene");
      show("scene", img_scene);
      setMouseCallback("scene", onMouseForAutomatically, 0);
      printf("=================\n");
    }

    if ((c == 'p') | (c == 'P'))
    {
        if(g_LeftTop_x >= g_RightBottom_x || g_LeftTop_y >= g_RightBottom_y)
        {
          printf("The rectangle selected is invalid\n");
          g_LeftTop_x = g_LeftTop_y = g_RightBottom_x = g_RightBottom_y = 0;
          show("scene", img_scene);
          continue;
        }

        CvRect RECT =cvRect(g_LeftTop_x, g_LeftTop_y, g_RightBottom_x-g_LeftTop_x, g_RightBottom_y-g_LeftTop_y);
        printf("=>lt_x: %d, lt_y: %d, w: %d, h: %d\n", g_LeftTop_x, g_LeftTop_y,  g_RightBottom_x-g_LeftTop_x, g_RightBottom_y-g_LeftTop_y);
        scene_splitted = img_scene(RECT).clone();
        findObjectInScene(img_object, scene_splitted);
    }

  }
  printf("=================\n");
  return 0;

MAUALLY_PROCESS:
  if( img_scene.empty() )
  {
    ReadMe();
    exit(-1);
  }

  Mat original_image = img_scene.clone();
  Mat image, warped_image;

  roi_corners.push_back(Point2f( (float)(original_image.cols/4), (float)(original_image.rows/4) ));
  roi_corners.push_back(Point2f( (float)(original_image.cols/4*3), (float)(original_image.rows/4*1) ));
  roi_corners.push_back(Point2f( (float)(original_image.cols/4*3), (float)(original_image.rows/4*3) ));
  roi_corners.push_back(Point2f( (float)(original_image.cols/4), (float)(original_image.rows/4*3) ));

  namedWindow(windowTitle, WINDOW_NORMAL);
  namedWindow("Warped Image", WINDOW_AUTOSIZE);
  moveWindow("Warped Image", 80, 20);
  resizeWindow(windowTitle, 960, 720);
  moveWindow(windowTitle, 800, 10);

  setMouseCallback(windowTitle, onMouseForManually, 0);

  bool endProgram = false;
  while (!endProgram)
  {
    if ( validation_needed & (roi_corners.size() < 4) )
    {
      validation_needed = false;
      image = original_image.clone();

      for (size_t i = 0; i < roi_corners.size(); ++i)
      {
        circle( image, roi_corners[i], 17, Scalar(0, 255, 0), 7 );

        if( i > 0 )
        {
          line(image, roi_corners[i-1], roi_corners[(i)], Scalar(0, 0, 255), 2);
          circle(image, roi_corners[i], 17, Scalar(0, 255, 0), 7);
          putText(image, labels[i].c_str(), roi_corners[i], QT_FONT_NORMAL, 2, Scalar(255, 0, 0), 5);
        }
      }
      imshow( windowTitle, image );
    }

    if ( validation_needed & ( roi_corners.size() == 4 ))
    {
      validation_needed = false;
      image = original_image.clone();
      for ( int i = 0; i < 4; ++i )
      {
          line(image, roi_corners[i], roi_corners[(i + 1) % 4], Scalar(0, 0, 255), 2);
          circle(image, roi_corners[i], 17, Scalar(0, 255, 0), 7);
          putText(image, labels[i].c_str(), roi_corners[i], QT_FONT_NORMAL, 3, Scalar(255, 0, 0), 5);
      }
      imshow( windowTitle, image );

      dst_corners[0].x = 0;
      dst_corners[0].y = 0;
      dst_corners[1].x = (float)std::max(norm(roi_corners[0] - roi_corners[1]), norm(roi_corners[2] - roi_corners[3]));
      dst_corners[1].y = 0;
      dst_corners[2].x = (float)std::max(norm(roi_corners[0] - roi_corners[1]), norm(roi_corners[2] - roi_corners[3]));
      dst_corners[2].y = (float)std::max(norm(roi_corners[1] - roi_corners[2]), norm(roi_corners[3] - roi_corners[0]));
      dst_corners[3].x = 0;
      dst_corners[3].y = (float)std::max(norm(roi_corners[1] - roi_corners[2]), norm(roi_corners[3] - roi_corners[0]));

      Size warped_image_size = Size(cvRound(dst_corners[2].x), cvRound(dst_corners[2].y));
      Mat H = findHomography(roi_corners, dst_corners); //get homography
      warpPerspective(original_image, warped_image, H, warped_image_size); // do perspective transformation
      show("Warped Image", warped_image);
    }

    char c = (char)waitKey( 10 );

    if ((c == 'q') | (c == 'Q') | (c == 27))
    {
      endProgram = true;
      char ts[32] = {0};
      getTimeStamp(ts);
      printf("\n%s ==> \n[%d, %d, %d, %d, %d, %d, %d, %d]\n", ts,
        (int)round((roi_corners[0]).x), (int)round((roi_corners[0]).y),
        (int)round((roi_corners[1]).x), (int)round((roi_corners[1]).y),
        (int)round((roi_corners[2]).x), (int)round((roi_corners[2]).y),
        (int)round((roi_corners[3]).x), (int)round((roi_corners[3]).y));

      if(!g_DebugEnable) continue;

      g_FileSaved = g_SceneFileName;
      size_t n = g_FileSaved.find_last_of(".", -1);
      g_FileSaved.insert(n, ts);
      imwrite(g_FileSaved, warped_image);
    }

    if ((c == 'h') | (c == 'H'))
    {
      roi_corners[selected_corner_index].x = roi_corners[selected_corner_index].x - 1;
      validation_needed = true;
    }

    if ((c == 'j') | (c == 'J'))
    {
      roi_corners[selected_corner_index].y = roi_corners[selected_corner_index].y + 1;
      validation_needed = true;
    }

    if ((c == 'k') | (c == 'K'))
    {
      roi_corners[selected_corner_index].y = roi_corners[selected_corner_index].y - 1;
      validation_needed = true;
    }

    if ((c == 'l') | (c == 'L'))
    {
      roi_corners[selected_corner_index].x = roi_corners[selected_corner_index].x + 1;
      validation_needed = true;
    }
  }

  printf("=================\n");
  return 0;
}

void onMouseForAutomatically( int event, int x, int y, int flags, void* param)
{
    if(g_InProgress != 0) return;

    // Action when left button is pressed
    if ( event == EVENT_LBUTTONDOWN)
    {
        g_LeftTop_x = x * ZOOM_IN_SCALE;
        g_LeftTop_y = y * ZOOM_IN_SCALE;
        g_LeftButtonIsDown = 1;
    }

    if ((event == EVENT_MOUSEMOVE) && g_LeftButtonIsDown)
    {
        g_RightBottom_x = x * ZOOM_IN_SCALE;
        g_RightBottom_y = y * ZOOM_IN_SCALE;

        img_scene_copy = img_scene.clone();
        rectangle(img_scene_copy, Point(g_LeftTop_x-LINE_WIDTH/2,g_LeftTop_y-LINE_WIDTH/2), Point(g_RightBottom_x+LINE_WIDTH/2,g_RightBottom_y+LINE_WIDTH/2),Scalar(255,255,255),LINE_WIDTH,1,0);
        show("scene", img_scene_copy);
    }

    // Action when left button is released
    if (event == EVENT_LBUTTONUP && g_LeftButtonIsDown)
    {
        g_LeftButtonIsDown = 0;
        g_RightBottom_x = x * ZOOM_IN_SCALE;
        g_RightBottom_y = y * ZOOM_IN_SCALE;
    }
}

void findObjectInScene(Mat img_object, Mat img_scene)
{
  g_InProgress = 1;

  //-- Step 1: Detect the keypoints using SURF Detector
  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  // printf("-- Max dist : %f \n", max_dist );
  // printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
    { good_matches.push_back( matches[i]); }
  }

  printf("Good Points: %d\n", good_matches.size());

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  //-- Localize the object from img_1 in img_2
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( size_t i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, CV_RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = Point(0,0); obj_corners[1] = Point( img_object.cols, 0 );
  obj_corners[2] = Point( img_object.cols, img_object.rows ); obj_corners[3] = Point( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);


  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  Point2f offset( (float)img_object.cols, 0);
  line( img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), LINE_WIDTH );
  line( img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar( 0, 255, 0), LINE_WIDTH );
  line( img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar( 0, 255, 0), LINE_WIDTH );
  line( img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar( 0, 255, 0), LINE_WIDTH );

  char ts[32] = {0};
  getTimeStamp(ts);
  printf("\n%s ==> \n[%d, %d, %d, %d, %d, %d, %d, %d]\n", ts,
    (int)round((scene_corners[0]).x) + g_LeftTop_x, (int)round((scene_corners[0]).y) + g_LeftTop_y,
    (int)round((scene_corners[1]).x) + g_LeftTop_x, (int)round((scene_corners[1]).y) + g_LeftTop_y,
    (int)round((scene_corners[2]).x) + g_LeftTop_x, (int)round((scene_corners[2]).y) + g_LeftTop_y,
    (int)round((scene_corners[3]).x) + g_LeftTop_x, (int)round((scene_corners[3]).y) + g_LeftTop_y);

  //-- Show detected matches
  show( "Good Matches & Object detection", img_matches );

  Mat warped_image;
  Size warped_image_size = Size(cvRound(obj_corners[2].x), cvRound(obj_corners[2].y));
  H = findHomography(scene_corners, obj_corners);
  warpPerspective(img_scene, warped_image, H, warped_image_size);

  if(g_DebugEnable)
  {
    g_FileSaved = g_SceneFileName;
    size_t n = g_FileSaved.find_last_of(".", -1);
    g_FileSaved.insert(n, ts);
    imwrite(g_FileSaved, warped_image);
  }
  show("Warped Image", warped_image);

  g_InProgress = 0;
}

void getTimeStamp(char* buf)
{
    time_t now;
    struct tm* tm_now;

    time(&now);
    tm_now = localtime(&now);
    snprintf(buf, 32, "@%d_%02d_%02d_%02d_%02d_%02d",  tm_now->tm_year+1900, tm_now->tm_mon, tm_now->tm_mday, tm_now->tm_hour, tm_now->tm_min, tm_now->tm_sec);
}

//////////////////////////////////////////////////////////////////////

void onMouseForManually( int event, int x, int y, int flags, void* param)
{
    // Action when left button is pressed
    if (roi_corners.size() == 4)
    {
        for (int i = 0; i < 4; ++i)
        {
            if ((event == EVENT_LBUTTONDOWN) & ((abs(roi_corners[i].x - x) < 20)) & (abs(roi_corners[i].y - y) < 20))
            {
                selected_corner_index = i;
                dragging = true;
            }
        }
    }
    else if ( event == EVENT_LBUTTONDOWN )
    {
        roi_corners.push_back( Point2f( (float) x, (float) y ) );
        validation_needed = true;
    }

    // Action when left button is released
    if (event == EVENT_LBUTTONUP)
    {
        dragging = false;
    }

    // Action when left button is pressed and mouse has moved over the window
    if ((event == EVENT_MOUSEMOVE) && dragging)
    {
        roi_corners[selected_corner_index].x = (float) x;
        roi_corners[selected_corner_index].y = (float) y;
        validation_needed = true;
    }
}
