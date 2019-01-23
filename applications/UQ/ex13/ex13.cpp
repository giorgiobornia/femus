#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "TransientSystem.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "NumericVector.hpp"
#include "adept.h"

#include "petsc.h"
#include "petscmat.h"
#include "PetscMatrix.hpp"

#include "slepceps.h"
#include "sparseGrid.hpp"


#include <vector>


using namespace femus;



//BEGIN stochastic data

bool sparse = true;

unsigned alpha = 4;
unsigned M = pow (10, alpha);   //number of samples
unsigned N = 2; //dimension of the parameter space (each of the M samples has N entries)
unsigned L = 4; //max refinement level
bool output = false; //for debugging
bool matlabView = true;

double xmin = - 5.5;   //-1.5 for uniform // -5.5 for Gaussian
double xmax = 5.5;     //1.5 for uniform // 5.5 for Gaussian

//FOR NORMAL DISTRIBUTION
boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
boost::normal_distribution<> nd (0., 1.);
boost::variate_generator < boost::mt19937&,
      boost::normal_distribution<> > var_nor (rng, nd);

//FOR UNIFORM DISTRIBUTION
boost::mt19937 rng1; // I don't seed it on purpouse (it's not relevant)
boost::random::uniform_real_distribution<> un (- 1., 1.);
boost::variate_generator < boost::mt19937&,
      boost::random::uniform_real_distribution<> > var_unif (rng1, un);

//FOR LAPLACE DISTRIBUTION
boost::mt19937 rng2; // I don't seed it on purpouse (it's not relevant)
boost::random::uniform_real_distribution<> un1 (- 0.5, 0.49999999999);
boost::variate_generator < boost::mt19937&,
      boost::random::uniform_real_distribution<> > var_unif1 (rng2, un1);
double b = 2.;
//END

int main (int argc, char** argv) {

  //BEGIN construction of the sample set
  std::vector < std::vector < double > >  samples;
  samples.resize (M);

  for (unsigned m = 0; m < M; m++) {
    samples[m].resize (N);
  }

  for (unsigned m = 0; m < M; m++) {
    for (unsigned n = 0; n < N; n++) {
      double var = var_nor();
      double varunif = var_unif();
      double U = var_unif1();
      samples[m][n] = var;
      //samples[m][n] = varunif;
    }
  }
  //END


  //Build Histogram
  unsigned dim = static_cast <unsigned> (pow (2, L - 1));
  std::vector < std::vector <double> > cI (dim);
  for (unsigned i = 0; i < dim; i++) cI[i].assign (dim, 0.);

  double h = (xmax - xmin) / dim;
  double h2 = h * h;

  for (unsigned m = 0; m < M; m++) {
    double x = (samples[m][0] - xmin) / h;
    unsigned i;
    if (x < 0.) i = 0;
    else if (x >= dim) i = dim - 1;
    else i = static_cast < unsigned > (floor (x));

    double y = (samples[m][1] - xmin) / h;
    unsigned j;
    if (y < 0.) j = 0;
    else if (y >= dim) j = dim - 1;
    else j = static_cast < unsigned > (floor (y));

    cI[i][j] += 1. / (M * h2);
  }

  
//   std::cout << "Histogram "<<std::endl;
//   for (unsigned i = 0; i < dim; i++) {
//     for (unsigned j = 0; j < dim; j++) {
//       std::cout << cI[i][j] << " ";
//     }
//     std::cout << std::endl;
//   }

  
  //Build Hierarchical Bases
   
  std::vector < std::vector < std::vector <std::vector <double> > > > cH (L);
  for (unsigned iL = 0; iL < cH.size(); iL++) {
    cH[iL].resize (L);
    unsigned iDim = static_cast <unsigned> (pow (2, iL));
    for (unsigned jL = 0; jL < cH[iL].size(); jL++) {
      cH[iL][jL].resize (iDim);
      unsigned jDim = static_cast <unsigned> (pow (2, jL));
      for (unsigned i = 0; i < cH[iL][jL].size(); i++) {
        cH[iL][jL][i].assign (jDim, 0.);
      }
    }
  }


  double Hx = (xmax - xmin);
  double Hy = (xmax - xmin);

  for (unsigned m = 0; m < M; m++) {
    double X = (samples[m][0] - xmin) / Hx;
    double Y = (samples[m][1] - xmin) / Hy;

    if (X < 0.)  X = 0.;
    if (Y >= 1.) X = 0.9999999999999999999999999999999999;

    if (Y < 0.)  Y = 0.;
    if (Y >= 1.) Y = 0.9999999999999999999999999999999999;

    for (unsigned iL = 0; iL < cH.size(); iL++) {
      double hx = Hx / pow (2., iL);
      double x = X * pow (2., iL);
      for (unsigned jL = 0; iL * sparse + jL < cH[iL].size(); jL++) {
        double hy = Hy / pow (2., jL);
        double y = Y * pow (2., jL);
        unsigned i = static_cast < unsigned > (floor (x));
        unsigned j = static_cast < unsigned > (floor (y));

        cH[iL][jL][i][j] += 1. / (M * hx * hy);
      }
    }
  }



  for (unsigned iL = 0; iL < cH.size(); iL++) {
    for (unsigned jL = 0; iL * sparse + jL < cH[iL].size(); jL++) {
      //std::cout << iL << " " << jL << std::endl;
      int iL1 = iL;
      while (iL1 >= 0) {
        int jL1 = (iL1 == iL) ? jL : jL + 1;
        while (jL1 >= 1) {
          jL1--;
          //std::cout << "\t\t" << iL1 << " " << jL1 << std::endl;
          for (unsigned  i = 0; i < cH[iL][jL].size(); i++) {
            unsigned i1 = static_cast < unsigned > (floor (i / pow (2., (iL - iL1))));
            for (unsigned  j = 0; j < cH[iL][jL][i].size(); j++) {
              unsigned j1 = static_cast < unsigned > (floor (j / pow (2., (jL - jL1))));
              cH[iL][jL][i][j] -= cH[iL1][jL1][i1][j1];
            }
          }
        }
        iL1--;
        //std::cout<<std::endl;
      }
    }
  }


//   std::cout << std::endl;
//   std::cout << "Hierarchical bases "<<std::endl;
//   for (unsigned iL = 0; iL < cH.size(); iL++) {
//     for (unsigned jL = 0; jL < cH[iL].size(); jL++) {
//       std::cout << "iL = " << iL << " jL = "<< jL <<std::endl;   
//       for (unsigned i = 0; i < cH[iL][iL].size(); i++) {
//         for (unsigned j = 0; j < cH[iL][jL][i].size(); j++) {
//           std::cout << cH[iL][jL][i][j] << " ";
//         }
//         std::cout << std::endl;
//       }
//       std::cout << std::endl;
//     }
//     std::cout << std::endl;
//   }


  
  //Reconstruct Histogram from Hierarchical Bases
  std::vector < std::vector <double> > cIr (dim);
  for (unsigned i = 0; i < dim; i++) cIr[i].assign (dim, 0.);

  for (unsigned i = 0; i < cIr.size(); i++) {
    for (unsigned j = 0; j < cIr[i].size(); j++) {
      int iL = L - 1;
      while (iL >= 0) {
        unsigned i1 = static_cast < unsigned > (floor (i / pow (2., (L - 1 - iL))));
        int jL = L - 1;
        while (jL >= 0) {
          unsigned j1 = static_cast < unsigned > (floor (j / pow (2., (L - 1 - jL))));
          cIr[i][j] += cH[iL][jL][i1][j1];
          jL--;
        }
        iL--;
      }
    }
  }

 // std::cout<<"Histogram reconstructed from Hierarchical Bases"<<std::endl;
  for (unsigned i = 0; i < dim; i++) {
    for (unsigned j = 0; j < dim; j++) {
      std::cout << cIr[i][j] << " ";
    }
    std::cout << std::endl;
  }
//   std::cout << std::endl;
//   std::cout << std::endl;
  
  

//   std::cout<<"Difference between Histogram and Histogram reconstructed from Hierarchical Bases"<<std::endl;
//   for (unsigned i = 0; i < dim; i++) {
//     for (unsigned j = 0; j < dim; j++) {
//       std::cout << (cI[i][j] - cIr[i][j]) << " ";
//     }
//     std::cout << std::endl;
//   }









  /*
    //BEGIN initialize grid and compute nodal values
    clock_t total_time = clock();
    clock_t grid_time = clock();
    sparseGrid spg (samples, xmin, xmax, output);

    std::cout << std::endl << " Builds sparse grid in: " << std::setw (11) << std::setprecision (6) << std::fixed
              << static_cast<double> ( (clock() - grid_time)) / CLOCKS_PER_SEC << " s" << std::endl;

    clock_t nodal_time = clock();
    spg.EvaluateNodalValuesPDF (samples);
    spg.PrintNodalValuesPDF();

    std::cout << std::endl << " Builds nodal values in: " << std::setw (11) << std::setprecision (6) << std::fixed
              << static_cast<double> ( (clock() - nodal_time)) / CLOCKS_PER_SEC << " s" << std::endl;
    //END



    //BEGIN  create grid for plot in 2D

    std::vector < unsigned > refinementLevel (N);

    refinementLevel[0] = L; //refinement level in x

    if (N > 1)  refinementLevel[1] = L;   //refinement level in y

    if (N > 2)  refinementLevel[2] = L;   //refinement level in x

    std::vector < unsigned > gridPoints (N);
    std::vector < std::vector < double> > gridBounds (N);

    for (unsigned n = 0; n < N; n++) {
      gridPoints[n] = static_cast<unsigned> (pow (2, refinementLevel[n]) + 1);
  //         gridPoints[n] = static_cast<unsigned> ( pow ( 2, refinementLevel[n] )  ); //to compare the histogram with the full grid
      gridBounds[n].resize (2);
    }

    unsigned gridSize = 1;

    for (unsigned n = 0; n < N; n++) {
      gridSize *= gridPoints[n];
    }

    gridBounds[0][0] = xmin;
    gridBounds[0][1] = xmax;

    if (N > 1) {

      gridBounds[1][0] = xmin;
      gridBounds[1][1] = xmax;
    }

    if (N > 2) {

      gridBounds[2][0] = xmin;
      gridBounds[2][1] = xmax;
    }

    std::vector < double > h (N);

    for (unsigned n = 0; n < N; n++) {
      h[n] = (gridBounds[n][1] - gridBounds[n][0]) / pow (2, refinementLevel[n]);
    }

    std::vector < std::vector < double > > grid;

    unsigned counterGrid = 0;

    if (N == 1) {
      for (unsigned i = 0; i < gridPoints[0]; i++) {
        grid.resize (counterGrid + 1);
        grid[counterGrid].resize (N);
        grid[counterGrid][0] = gridBounds[0][0] + i * h[0];
        counterGrid++;
      }
    }

    else if (N > 1) {
      for (unsigned j = 0; j < gridPoints[1]; j++) {
        for (unsigned i = 0; i < gridPoints[0]; i++) {
          grid.resize (counterGrid + 1);
          grid[counterGrid].resize (N);
          grid[counterGrid][0] = gridBounds[0][0] + i * h[0];
          grid[counterGrid][1] = gridBounds[1][0] + j * h[1];
  //                 grid[counterGrid][0] = ( gridBounds[0][0] + h[0] * 0.5 ) + i * h[0]; //to compare the histogram with the full grid
  //                 grid[counterGrid][1] = ( gridBounds[1][0] + h[1] * 0.5 ) + j * h[0]; //to compare the histogram with the full grid
          counterGrid++;
        }
      }
    }

    //END create grid


    //BEGIN create histogram on finest grid for comparison
    unsigned histoSize1D =  static_cast<unsigned> (pow (2, refinementLevel[0])) ;
    unsigned totNumberOfBins = static_cast<unsigned> (pow (histoSize1D, N));     //tot number of bins
    std::vector <double> histogram (totNumberOfBins);

    std::vector < std::vector < double > > histoGrid (totNumberOfBins);

    for (unsigned i = 0; i < histoGrid.size(); i++) {
      histoGrid[i].resize (N);
    }

    unsigned counterHisto = 0;

    if (N == 1) {
      for (unsigned i = 0; i < histoGrid.size(); i++) {
        histoGrid[counterHisto][0] = (gridBounds[0][0] + h[0] * 0.5) + i * h[0];
        counterHisto++;
      }
    }

    else if (N > 1) {
      for (unsigned j = 0; j < histoSize1D; j++) {
        for (unsigned i = 0; i < histoSize1D; i++) {
          histoGrid[counterHisto][0] = (gridBounds[0][0] + h[0] * 0.5) + i * h[0];
          histoGrid[counterHisto][1] = (gridBounds[1][0] + h[1] * 0.5) + j * h[1];
          counterHisto++;
        }
      }
    }

    for (unsigned m = 0; m < M; m++) {
      for (unsigned i = 0; i < histoGrid.size(); i++) {

        std::vector<unsigned> inSupport (N, 0);

        for (unsigned n = 0; n < N; n++) {
          if (samples[m][n] > histoGrid[i][n] - h[n] && samples[m][n] <= histoGrid[i][n] + h[n]) {
            inSupport[n] = 1;
          }
        }

        unsigned sumCheck = 0;

        for (unsigned n = 0; n < N; n++) {
          sumCheck += inSupport[n];
        }

        if (sumCheck == N) {
          histogram[i]++;
          break;
        }

      }
    }

    //evaluation of histogram integral

    double histoIntegral = 0.;
    double supportMeasure = pow (h[0], N);

    for (unsigned i = 0; i < histoGrid.size(); i++) {
      histoIntegral += supportMeasure * histogram[i];
    }

    for (unsigned i = 0; i < histoGrid.size(); i++) {
      histogram[i] /= histoIntegral;
    }

    std::cout << " HISTO" << std::endl;
    for (unsigned i = 0; i < histoGrid.size(); i++) {
      for (unsigned n = 0; n < N; n++) {
        std::cout << histoGrid[i][n] << " , " ;
      }

      std::cout << histogram[i] << std::endl;
    }


  //END



  //BEGIN grid plot
    if (matlabView) {
      std::cout << "x=[" << std::endl;

      for (unsigned i = 0; i < grid.size(); i++) {
        std::cout << grid[i][0] << std::endl;
      }

      std::cout << "];" << std::endl;

      if (N > 1) {
        std::cout << "y=[" << std::endl;

        for (unsigned i = 0; i < grid.size(); i++) {
          std::cout << grid[i][1] << std::endl;
        }

        std::cout << "];" << std::endl;
      }

      clock_t pdf_time = clock();

      std::cout << "PDF=[" << std::endl;

      for (unsigned i = 0; i < grid.size(); i++) {
        double pdfValue;
        spg.EvaluatePDF (pdfValue, grid[i], true);
      }

      std::cout << "];" << std::endl;

      std::cout << std::endl << " Builds PDF in: " << std::setw (11) << std::setprecision (6) << std::fixed
                << static_cast<double> ( (clock() - pdf_time)) / CLOCKS_PER_SEC << " s" << std::endl;

    }


  //END grid plot

    double integral;
    spg.EvaluatePDFIntegral (integral);

    std::cout << " PDF Integral is = " << integral << std::endl;


  //BEGIN compute error
    clock_t error_time = clock();

    double aL2E;
    spg.ComputeAvgL2Error (aL2E, samples, 1);

    std::cout << " Averaged L2 error is = " << aL2E << std::endl;

    std::cout << " Computes error in: " << std::setw (11) << std::setprecision (6) << std::fixed
              << static_cast<double> ( (clock() - error_time)) / CLOCKS_PER_SEC << " s" << std::endl;
  //END
    std::cout << " Total time: " << std::setw (11) << std::setprecision (6) << std::fixed
              << static_cast<double> ( (clock() - total_time)) / CLOCKS_PER_SEC << " s" << std::endl;*/

  return 0;

} //end main



