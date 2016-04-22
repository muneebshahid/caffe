#include <algorithm>
#include <vector>
#include <numeric>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/layers/dotproduct_similarity_layer.hpp"

namespace caffe{

template <typename TypeParam>

class DotProductSimilarityLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  
  protected:
  DotProductSimilarityLayerTest()
      : blob_bottom_data_i_(new Blob<Dtype>(32, 512, 1, 1)),
        blob_bottom_data_j_(new Blob<Dtype>(32, 512, 1, 1)),
        blob_top(new Blob<Dtype>()) {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    blob_bottom_vec_.push_back(blob_bottom_data_i_);
    blob_bottom_vec_.push_back(blob_bottom_data_j_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~DotProductSimilarityLayerTest() {
    delete blob_bottom_data_i_;
    delete blob_bottom_data_j_;
    delete blob_top;
  }

  void TestForward()
  {
  	FillerParameter filler_param;
  	UniformFiller<Dtype> filler(filler_param);
  	filler.Fill(this->blob_bottom_data_i_);
  	filler.Fill(this->blob_bottom_data_j_);
  	
  	LayerParameter layer_param;
  	DotProductSimilarityLayer<Dtype> layer(layer_param);
  	layer.SetUp(this->blob_bottom_vec, this->blob_top_vec_);
  	layer.Forward(this->blob_bottom_vec, this->blob_top_vec_);
  	const Dtype precision = 1e-5;
  	  int dim = this->blob_bottom_vec[0]->count() / this->blob_bottom_vec[0]->num();
	  for (int n = 0; n < this->blob_bottom_vec[0]->num(); ++n) {
	  	float vec_1[dim] = {};
	  	float vec_2[dim] = {};
	    for (int c = 0; c < this->blob_top_->channels(); ++c) {
	      for (int h = 0; h < this->blob_top_->height(); ++h) {
	        for (int w = 0; w < this->blob_top_->width(); ++w) {
	        	vec_1[c + h + w] = this->blob_bottom_vec[0]->data_at(n, c, h, w);
	        	vec_2[c + h + w] = this->blob_bottom_vec[0]->data_at(n, c, h, w);	          
	        }
	      }
	    }
	    float result = std::inner_product(vec_1, vec_1 + dim, vec_2, 0);
	    EXPECT_NEAR(result, this->blob_top_->data_at(n, 1, 1, 1), precision);
	  }
  }

  void TestBackward()
  {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_i_);
    filler.Fill(this->blob_bottom_data_j_);

    LayerParameter layer_param;
    DotProductSimilarityLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  TYPED_TEST_CASE(DotProductSimilarityLayerTest, TestDtypesAndDevices);
  
  TYPED_TEST(DotProductSimilarityLayerTest, TestForward) 
  {
  	this->TestForward();
  }

  TYPED_TEST(DotProductSimilarityLayerTest, TestBackward) 
  {
  	this->TestBackward();
  }

  Blob<Dtype>* const blob_bottom_data_i_;
  Blob<Dtype>* const blob_bottom_data_j_;
  Blob<Dtype>* const blob_top;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
 };
}