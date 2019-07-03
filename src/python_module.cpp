#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

namespace pbcvt {

    using namespace boost::python;

    PyObject *tvl1_optical_flow_cuda(PyObject *left, PyObject *right) {

        cv::Mat leftMat, rightMat;
        leftMat = pbcvt::fromNDArrayToMat(left);
        rightMat = pbcvt::fromNDArrayToMat(right);

        if (leftMat.empty())
        {
            std::cerr << "Can't open image."<< std::endl;
            //return -1;
        }
        if (rightMat.empty())
        {
            std::cerr << "Can't open image."<< std::endl;
            //return -1;
        }

        if (leftMat.size() != rightMat.size())
        {
            std::cerr << "Images should be of equal sizes" << std::endl;
            //return -1;
        }
        ///////////////////////////////////////////////
        //cv::GpuMat d_frame0,d_frame1;
        cv::cuda::GpuMat d_frame0(leftMat);
        cv::cuda::GpuMat d_frame1(rightMat);

        //cv::cuda::GpuMat resized_frame0(224,224,CV_32FC2);
        //cd::cuda::GpuMat resized_frame1(224,224,CV_32FC2);
        //cv::cuda::resize(d_frame0,resized_frame0,Size(224,224),0,0,INTER_CUBIC);
        //cv::Mat resized_frame0_cpu(resized_frame0);
        cv::cuda::GpuMat d_flow;


        Ptr<cv::cuda::OpticalFlowDual_TVL1> tvl1 = cv::cuda::OpticalFlowDual_TVL1::create();
        {
        const int64 start = getTickCount();

        tvl1->calc(d_frame0, d_frame1, d_flow);

        const double timeSec = (getTickCount() - start) / getTickFrequency();
        std::cout << "TVL1 : " << timeSec << " sec" << std::endl;
        

       // showFlow("TVL1", d_flow);
        }

    //GpuMat d_flow(224,224, CV_32FC2);
        //cv::cuda::GpuMat d_flow(leftMat.size(),CV_32FC2);

        cv::Mat d_flow_cpu(d_flow);

        ///////////////////////////////////////////////
        //cv::Mat result = leftMat * rightMat;
        PyObject *ret = pbcvt::fromMatToNDArray(d_flow_cpu);
        return ret;
    }

#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }

    BOOST_PYTHON_MODULE (pbcvt) {
        //using namespace XM;
        init_ar();

        //initialize converters
        to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
        matFromNDArrayBoostConverter();

        //expose module-level functions
        def("tvl1_optical_flow_cuda", tvl1_optical_flow_cuda);
        //def("dot2", dot2);
		//def("makeCV_16UC3Matrix", makeCV_16UC3Matrix);

		//from PEP8 (https://www.python.org/dev/peps/pep-0008/?#prescriptive-naming-conventions)
        //"Function names should be lowercase, with words separated by underscores as necessary to improve readability."
        //def("increment_elements_by_one", increment_elements_by_one);
    }

} //end namespace pbcvt