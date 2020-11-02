#include <iostream>
#include "TemplatedDatabase.h"
#include "TemplatedVocabulary.h"
#include "BowVector.h"
#include "FeatureVector.h"
#include "QueryResults.h"
#include "FORB.h"
#include "FSift.h"
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp> // sift

using namespace DBoW2;
using namespace std;

/// SIFT Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FSift::TDescriptor, DBoW2::FSift> 
  SiftVocabulary;

/// SIFT Database
typedef DBoW2::TemplatedDatabase<DBoW2::FSift::TDescriptor, DBoW2::FSift> 
  SiftDatabase;

void loadFeatures(vector<vector<vector<float > > > &features);
void changeStructure(const cv::Mat &plain, vector<vector<float > > &out);
void testVocCreation(const vector<vector<vector<float > > > &features);
void testDatabase(const vector<vector<vector<float > > > &features);

// number of training images
const int NIMAGES = 4;

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

int main()
{
  vector<vector<vector<float > > > features;
  loadFeatures(features);

  testVocCreation(features);

  wait();

  testDatabase(features);

  return 0;
}

void loadFeatures(vector<vector<vector<float > > > &features)
{
    features.clear();
    features.reserve(NIMAGES);
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(1000);
    cout << "Rxtracting SIFT features..." << endl;
    for(int i = 0; i < NIMAGES; ++i)
    {
        stringstream ss;
        ss << "images/image" << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        sift -> detectAndCompute(image, mask, keypoints, descriptors);
        
        features.push_back(vector<vector<float > >());
        changeStructure(descriptors, features.back());
    }

}

void changeStructure(const cv::Mat &plain, vector<vector<float > > &out)
{
    out.resize(plain.rows);
    for(int j = 0; j < plain.rows; ++j)
    {
        out[j].reserve(plain.cols);
        for(int i = 0; i < plain.cols; i++)
        {
            out[j].push_back(plain.at<float>(j, i));
        }
    }
}

void testVocCreation(const vector<vector<vector<float > > > &features)
{
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType scoring = L2_NORM;

    SiftVocabulary voc(k, L, weight, scoring);
    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);   
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    for(int i = 0; i < NIMAGES; i++)
    {
        voc.transform(features[i], v1);
        for(int j = 0; j < NIMAGES; j++)
        {
        voc.transform(features[j], v2);
        
        double score = voc.score(v1, v2);
        cout << "Image " << i << " vs Image " << j << ": " << score << endl;
        }
    }

    // save the vocabulary to disk
    cout << endl << "Saving sift vocabulary..." << endl;
    voc.save("small_sift_voc.yml.gz");
    cout << "Done" << endl;
}

void testDatabase(const vector<vector<vector<float > > > &features)
{
    cout << "Creating a small database..." << endl;
    SiftVocabulary voc("small_sift_voc.yml.gz");
    SiftDatabase db(voc, false, 0);

    for(int i = 0; i < NIMAGES; i++)
    {
        db.add(features[i]);
    }
    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;
  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_sift_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  SiftDatabase db2("small_sift_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;

}