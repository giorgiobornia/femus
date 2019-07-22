/* This example details the full implementation of the p-Willmore flow
 *   algorithm, which involves three nonlinear systems.
 *
 *   System0 AssembleInit computes the initial curvatures given mesh positions.
 *   System AssemblePWillmore solves the flow equations.
 *   System2 AssembleConformalMinimization "reparametrizes" the surface to
 *   correct the mesh. */

#include "FemusInit.hpp"
#include "MultiLevelSolution.hpp"
#include "MultiLevelProblem.hpp"
#include "NumericVector.hpp"
#include "VTKWriter.hpp"
#include "GMVWriter.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "TransientSystem.hpp"
#include "adept.h"
#include <cstdlib>
#include "petsc.h"
#include "petscmat.h"
#include "PetscMatrix.hpp"

using namespace femus;

void AssembleConformalMinimization (MultiLevelProblem&);  //stable and not bad
void AssembleShearMinimization (MultiLevelProblem&);  //vastly inferior

// IBVs.  No boundary, and IVs set to sphere (just need something).
bool SetBoundaryCondition (const std::vector < double >& x, const char solName[], double& value, const int faceName, const double time) {
  
  bool dirichlet = true;
  
  if(!strcmp(solName, "Dx1")) {
    if(1 == faceName || 3 == faceName) {
      dirichlet = false;
    }
  }
  else if(!strcmp(solName, "Dx2")) {
    if(2 == faceName || 4 == faceName) {
      dirichlet = false;
    }
  }
  
  
  value = 0.;
  return dirichlet;
}


// Main program starts here.
int main (int argc, char** args) {

  // init Petsc-MPI communicator
  FemusInit mpinit (argc, args, MPI_COMM_WORLD);


  // define multilevel mesh
  unsigned maxNumberOfMeshes;
  MultiLevelMesh mlMsh;

  // Read coarse level mesh and generate finer level meshes.
  double scalingFactor = 1.;

  //mlMsh.ReadCoarseMesh("../input/torus.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/sphere.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/ellipsoidRef3.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/ellipsoidV1.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/genusOne.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/knot.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/cube.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/horseShoe.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/tiltedTorus.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/dog.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/virus3.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh ("../input/ellipsoidSphere.neu", "seventh", scalingFactor);
  //mlMsh.ReadCoarseMesh("../input/CliffordTorus.neu", "seventh", scalingFactor);

  mlMsh.ReadCoarseMesh ("../input/square.neu", "seventh", scalingFactor);

  // Set number of mesh levels.
  unsigned numberOfUniformLevels = 3;
  unsigned numberOfSelectiveLevels = 0;
  mlMsh.RefineMesh (numberOfUniformLevels , numberOfUniformLevels + numberOfSelectiveLevels, NULL);

  // Erase all the coarse mesh levels.
  mlMsh.EraseCoarseLevels (numberOfUniformLevels - 1);

  // print mesh info
  mlMsh.PrintInfo();

  // Define the multilevel solution and attach the mlMsh object to it.
  MultiLevelSolution mlSol (&mlMsh);

  // Add variables X,Y,W to mlSol.
  mlSol.AddSolution ("Dx1", LAGRANGE, FIRST, 0);
  mlSol.AddSolution ("Dx2", LAGRANGE, FIRST, 0);
  mlSol.AddSolution ("Dx3", LAGRANGE, FIRST, 0);


  // Initialize the variables and attach boundary conditions.
  mlSol.Initialize ("All");

  mlSol.AttachSetBoundaryConditionFunction (SetBoundaryCondition);
  mlSol.GenerateBdc ("All");

  MultiLevelProblem mlProb (&mlSol);

  // Add system Conformal Minimization in mlProb.
  NonLinearImplicitSystem& system = mlProb.add_system < NonLinearImplicitSystem > ("conformal");

  // Add solutions newDX, Lambda1 to system.
  system.AddSolutionToSystemPDE ("Dx1");
  system.AddSolutionToSystemPDE ("Dx2");
  system.AddSolutionToSystemPDE ("Dx3");

  // Parameters for convergence and # of iterations.
  system.SetMaxNumberOfNonLinearIterations (40);
  system.SetNonLinearConvergenceTolerance (1.e-10);

  // Attach the assembling function to system and initialize.
  system.SetAssembleFunction (AssembleConformalMinimization);
  system.init();

  mlSol.SetWriter (VTK);
  std::vector<std::string> mov_vars;
  mov_vars.push_back ("Dx1");
  mov_vars.push_back ("Dx2");
  mlSol.GetWriter()->SetMovingMesh (mov_vars);

  // and this?
  std::vector < std::string > variablesToBePrinted;
  variablesToBePrinted.push_back ("All");
  mlSol.GetWriter()->SetDebugOutput (true);
  mlSol.GetWriter()->Write (DEFAULT_OUTPUTDIR, "linear", variablesToBePrinted, 0);

  system.MGsolve();

  mlSol.GetWriter()->Write (DEFAULT_OUTPUTDIR, "linear", variablesToBePrinted, 1);

  return 0;
}



// Building the Conformal Minimization system.
void AssembleConformalMinimization (MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled

  // call the adept stack object
  adept::Stack& s = FemusInit::_adeptStack;

  //  Extract pointers to the several objects that we are going to use.
  NonLinearImplicitSystem* mlPdeSys   = &ml_prob.get_system< NonLinearImplicitSystem> ("conformal");   // pointer to the linear implicit system named "Poisson"

  const unsigned level = mlPdeSys->GetLevelToAssemble();

  // Pointers to the mesh (level) object and elem object in mesh (level).
  Mesh *msh = ml_prob._ml_msh->GetLevel (level);
  elem *el = msh->el;

  // Pointers to the multilevel solution, solution (level) and equation (level).
  MultiLevelSolution *mlSol = ml_prob._ml_sol;
  Solution *sol = ml_prob._ml_sol->GetSolutionLevel (level);
  LinearEquationSolver *pdeSys = mlPdeSys->_LinSolver[level];

  // Pointers to global stiffness matrix and residual vector in pdeSys (level).
  SparseMatrix *KK = pdeSys->_KK;
  NumericVector *RES = pdeSys->_RES;

  // Convenience variables to keep track of the dimension.
  const unsigned  dim = 2;
  const unsigned  DIM = 3;

  // Get the process_id (for parallel computation).
  unsigned iproc = msh->processor_id();

  // Setting the reference elements to be equilateral triangles.
  std::vector < std::vector < double > > xT (2);
  xT[0].resize (3);
  xT[0][0] = -0.5;
  xT[0][1] = 0.5;
  xT[0][2] = 0.;

  xT[1].resize (3);
  xT[1][0] = 0.;
  xT[1][1] = 0.;
  xT[1][2] = sqrt (3.) / 2.;

  std::vector<double> phi_uv0;
  std::vector<double> phi_uv1;

  std::vector< double > stdVectorPhi;
  std::vector< double > stdVectorPhi_uv;

  // Extract positions of Dx in ml_sol object.
  unsigned solDxIndex[DIM];
  solDxIndex[0] = mlSol->GetIndex ("Dx1");
  solDxIndex[1] = mlSol->GetIndex ("Dx2");
  solDxIndex[2] = mlSol->GetIndex ("Dx3");

  // Extract finite element type for the solution.
  unsigned solType;
  solType = mlSol->GetSolutionType (solDxIndex[0]);

  // Local solution vectors for X, Dx, Xhat, XC.
  //std::vector < double > solx[DIM];
  //std::vector < double > solDx[DIM];
  //std::vector < double > xhat[DIM];
  //std::vector < double > xc[DIM];

  // Get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC).
  unsigned xType = 2;

  // Get the poitions of Y in the ml_sol object.
  //unsigned solNDxIndex[DIM];
  //solNDxIndex[0] = mlSol->GetIndex ("nDx1");
  //solNDxIndex[1] = mlSol->GetIndex ("nDx2");
  //solNDxIndex[2] = mlSol->GetIndex ("nDx3");

  // Get the positions of Y in the pdeSys object.
  unsigned solDxPdeIndex[DIM];
  solDxPdeIndex[0] = mlPdeSys->GetSolPdeIndex ("Dx1");
  solDxPdeIndex[1] = mlPdeSys->GetSolPdeIndex ("Dx2");
  solDxPdeIndex[2] = mlPdeSys->GetSolPdeIndex ("Dx3");

  // Local solution vectors for Nx and NDx.
  std::vector < adept::adouble > solDx[DIM];
  std::vector < adept::adouble > solx[DIM];

  // Get the position of "Lambda1" in the ml_sol object.
  //unsigned solLIndex;
  //solLIndex = mlSol->GetIndex ("Lambda1");

  // Get the finite element type for "Lambda1".
  //unsigned solLType;
  //solLType = mlSol->GetSolutionType (solLIndex);

  // Get the position of "Lambda1" in the pdeSys object.
  //unsigned solLPdeIndex;
  //solLPdeIndex = mlPdeSys->GetSolPdeIndex ("Lambda1");

  // Local Lambda1 solution.
  //std::vector < adept::adouble > solL;

  // Local-to-global pdeSys dofs.
  std::vector < int > SYSDOF;

  // Local residual vectors.
  vector< double > Res;
  std::vector< adept::adouble > aResDx[DIM];
// std::vector< adept::adouble > aResL;

  // Local Jacobian matrix (ordered by column).
  vector < double > Jac;

  KK->zero();  // Zero all the entries of the Global Matrix
  RES->zero(); // Zero all the entries of the Global Residual

  // ELEMENT LOOP: each process loops only on the elements that it owns.
  for (int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

    // Numer of solution element dofs.
    short unsigned ielGeom = msh->GetElementType (iel);
    unsigned nxDofs  = msh->GetElementDofNumber (iel, solType);
    //unsigned nLDofs  = msh->GetElementDofNumber (iel, solLType);

    // Resize local arrays.
    for (unsigned K = 0; K < DIM; K++) {
      solDx[K].resize (nxDofs);
      solx[K].resize (nxDofs);
    }

    // Resize local arrays
    SYSDOF.resize (DIM * nxDofs);
    Res.resize (DIM * nxDofs);

    for (unsigned K = 0; K < DIM; K++) {
      aResDx[K].assign (nxDofs, 0.);
    }

    // local storage of global mapping and solution
    for (unsigned i = 0; i < nxDofs; i++) {
      // Global-to-local mapping between X solution node and solution dof.
      unsigned iDDof = msh->GetSolutionDof (i, iel, solType);
      for (unsigned K = 0; K < DIM; K++) {
        solDx[K][i] = (*sol->_Sol[solDxIndex[K]]) (iDDof);
        // Global-to-global mapping between NDx solution node and pdeSys dof.
        SYSDOF[ K * nxDofs + i] =
          pdeSys->GetSystemDof (solDxIndex[K], solDxPdeIndex[K], i, iel);
      }
    }

    // start a new recording of all the operations involving adept variables.
    s.new_recording();
    for (unsigned i = 0; i < nxDofs; i++) {
      unsigned iXDof  = msh->GetSolutionDof (i, iel, xType);
      for (unsigned K = 0; K < DIM; K++) {
        solx[K][i] = (*msh->_topology->_Sol[K]) (iXDof) + solDx[K][i];
      }
    }

    // *** Gauss point loop ***
    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solType]->GetGaussPointNumber(); ig++) {

      const double *phix;  // local test function
      //const double *phiL;  // local test function
      const double *phix_uv[dim]; // local test function first order partial derivatives

      double weight; // gauss point weight

      // Get Gauss point weight, test function, and first order derivatives.
      if (ielGeom == QUAD) {
        phix = msh->_finiteElement[ielGeom][solType]->GetPhi (ig);
        //phiL = msh->_finiteElement[ielGeom][solLType]->GetPhi (ig);

        phix_uv[0] = msh->_finiteElement[ielGeom][solType]->GetDPhiDXi (ig);
        phix_uv[1] = msh->_finiteElement[ielGeom][solType]->GetDPhiDEta (ig);

        weight = msh->_finiteElement[ielGeom][solType]->GetGaussWeight (ig);
      }

      // Special adjustments for triangles.
//       else {
//         msh->_finiteElement[ielGeom][solType]->Jacobian (xT, ig, weight, stdVectorPhi, stdVectorPhi_uv);
// 
//         phix = &stdVectorPhi[0];
// 
//         phi_uv0.resize (nxDofs);
//         phi_uv1.resize (nxDofs);
// 
// 
//         for (unsigned i = 0; i < nxDofs; i++) {
//           phi_uv0[i] = stdVectorPhi_uv[i * dim];
//           phi_uv1[i] = stdVectorPhi_uv[i * dim + 1];
//         }
// 
//         phix_uv[0] = &phi_uv0[0];
//         phix_uv[1] = &phi_uv1[0];
// 
//         //phiL = msh->_finiteElement[ielGeom][solLType]->GetPhi (ig);
// 
//       }

      // Initialize and compute values of x, Dx, NDx, x_uv at the Gauss points.
      //double solDxg[3] = {0., 0., 0.};
      adept::adouble solDxg[3] = {0., 0., 0.};

      //double solx_uv[3][2] = {{0., 0.}, {0., 0.}, {0., 0.}};
      adept::adouble solx_uv[3][2] = {{0., 0.}, {0., 0.}, {0., 0.}};

      for (unsigned K = 0; K < DIM; K++) {
        for (unsigned i = 0; i < nxDofs; i++) {
          solDxg[K] += phix[i] * solDx[K][i];
          //solNDxg[K] += phix[i] * solNDx[K][i];
        }
        for (int j = 0; j < dim; j++) {
          for (unsigned i = 0; i < nxDofs; i++) {
            //solx_uv[K][j]    += phix_uv[j][i] * solx[K][i];
            solx_uv[K][j]   += phix_uv[j][i] * solx[K][i];
          }
        }
      }

      ///////// ADDED THIS /////////
      //adept::adouble solLg = 0.;
      //for (unsigned i = 0; i < nLDofs; i++) {
      //solLg += phiL[i] * solL[i];
      //}

      // Compute the metric, metric determinant, and area element.
      adept::adouble g[dim][dim] = {{0., 0.}, {0., 0.}};
      for (unsigned i = 0; i < dim; i++) {
        for (unsigned j = 0; j < dim; j++) {
          for (unsigned K = 0; K < DIM; K++) {
            g[i][j] += solx_uv[K][i] * solx_uv[K][j];
          }
        }
      }
    
      adept::adouble detg = g[0][0] * g[1][1] - g[0][1] * g[1][0];
      adept::adouble Area = weight * sqrt (detg);
      double Area2 = weight; // Trick to give equal weight to each element.

      // Compute components of the unit normal N.
      double normal[DIM];
      //normal[0] = (solx_uv[1][0] * solx_uv[2][1] - solx_uv[2][0] * solx_uv[1][1]) / sqrt (detg);
      //normal[1] = (solx_uv[2][0] * solx_uv[0][1] - solx_uv[0][0] * solx_uv[2][1]) / sqrt (detg);
      //normal[2] = (solx_uv[0][0] * solx_uv[1][1] - solx_uv[1][0] * solx_uv[0][1]) / sqrt (detg);

      normal[0] = 0.;
      normal[1] = 0.;
      normal[2] = 1.;

      // Discretize the equation \delta CD = 0 on the basis d/du, d/dv.
      adept::adouble V[DIM];
      V[0] = solx_uv[0][1] - normal[1] * solx_uv[2][0] + normal[2] * solx_uv[1][0];
      V[1] = solx_uv[1][1] - normal[2] * solx_uv[0][0] + normal[0] * solx_uv[2][0];
      V[2] = solx_uv[2][1] - normal[0] * solx_uv[1][0] + normal[1] * solx_uv[0][0];

      adept::adouble W[DIM];
      W[0] = solx_uv[0][0] + normal[1] * solx_uv[2][1] - normal[2] * solx_uv[1][1];
      W[1] = solx_uv[1][0] + normal[2] * solx_uv[0][1] - normal[0] * solx_uv[2][1];
      W[2] = solx_uv[2][0] + normal[0] * solx_uv[1][1] - normal[1] * solx_uv[0][1];

      adept::adouble M[DIM][dim];
      M[0][0] = W[0] - normal[2] * V[1] + normal[1] * V[2];
      M[1][0] = W[1] - normal[0] * V[2] + normal[2] * V[0];
      M[2][0] = W[2] - normal[1] * V[0] + normal[0] * V[1];

      M[0][1] = V[0] + normal[2] * W[1] - normal[1] * W[2];
      M[1][1] = V[1] + normal[0] * W[2] - normal[2] * W[0];
      M[2][1] = V[2] + normal[1] * W[0] - normal[0] * W[1];

//       // Compute new X minus old X dot N, for "reparametrization".
//       adept::adouble DnXmDxdotN = 0.;
//       for (unsigned K = 0; K < DIM; K++) {
//         DnXmDxdotN += (solDxg[K] - solNDxg[K]) * normal[K];
//       }

//       // Compute the metric inverse.
//       adept::adouble gi[dim][dim];
//       gi[0][0] =  g[1][1] / detg;
//       gi[0][1] = -g[0][1] / detg;
//       gi[1][0] = -g[1][0] / detg;
//       gi[1][1] =  g[0][0] / detg;
// 
//       // Compute the "reduced Jacobian" g^{ij}X_j .
//       double Jir[2][3] = {{0., 0., 0.}, {0., 0., 0.}};
//       for (unsigned i = 0; i < dim; i++) {
//         for (unsigned J = 0; J < DIM; J++) {
//           for (unsigned k = 0; k < dim; k++) {
//             Jir[i][J] += gi[i][k] * solx_uv[J][k];
//           }
//         }
//       }

      // Implement the Conformal Minimization equations.
      for (unsigned K = 0; K < DIM; K++) {
        for (unsigned i = 0; i < nxDofs; i++) {
          adept::adouble term1 = 0.;

          for (unsigned j = 0; j < dim; j++) {
            term1 +=  M[K][j] * phix_uv[j][i];
            //term1 +=  X_uv[K][j] * phix_uv[j][i];
          }

          // Conformal energy equation (with trick).
          aResDx[K][i] += term1 * Area2;
                           //+ timederiv * (solNDxg[K] - solDxg[K]) * phix[i] * Area2
                           //+ solL[0] * phix[i] * normal[K] * Area;
        }
      }

      // Lagrange multiplier equation (with trick).
//       for (unsigned i = 0; i < nLDofs; i++) {
//         //aResL[i] += phiL[i] * (DnXmDxdotN + eps * solL[i]) * Area; // no2
//       }
      //aResL[0] += (DnXmDxdotN + eps * solL[0]) * Area;

    } // end GAUSS POINT LOOP

    //------------------------------------------------------------------------
    // Add the local Matrix/Vector into the global Matrix/Vector
    //copy the value of the adept::adoube aRes in double Res and store

    for (int K = 0; K < DIM; K++) {
      for (int i = 0; i < nxDofs; i++) {
        Res[ K * nxDofs + i] = -aResDx[K][i].value();
      }
    }

//     for (int i = 0; i < nLDofs; i++) {
//       Res[DIM * nxDofs + i] = - aResL[i].value();
//     }
    RES->add_vector_blocked (Res, SYSDOF);

    // Resize Jacobian.
    Jac.resize ( (DIM * nxDofs) * (DIM * nxDofs));

    // Define the dependent variables.
    for (int K = 0; K < DIM; K++) {
      s.dependent (&aResDx[K][0], nxDofs);
    }
   // s.dependent (&aResL[0], nLDofs);


    // Define the independent variables.
    for (int K = 0; K < DIM; K++) {
      s.independent (&solDx[K][0], nxDofs);
    }
    //s.independent (&solL[0], nLDofs);

    // Get the jacobian matrix (ordered by row).
    s.jacobian (&Jac[0], true);

    KK->add_matrix_blocked (Jac, SYSDOF, SYSDOF);

    s.clear_independents();
    s.clear_dependents();

  } //end ELEMENT LOOP for each process.

  RES->close();
  KK->close();

} // end AssembleConformalMinimization.


void AssembleShearMinimization (MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled
  //  levelMax is the Maximum level of the MultiLevelProblem
  //  assembleMatrix is a flag that tells if only the residual or also the matrix should be assembled

  // call the adept stack object
  adept::Stack& s = FemusInit::_adeptStack;

  //  extract pointers to the several objects that we are going to use
  NonLinearImplicitSystem* mlPdeSys   = &ml_prob.get_system< NonLinearImplicitSystem> ("nProj");   // pointer to the linear implicit system named "Poisson"

  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh *msh = ml_prob._ml_msh->GetLevel (level);   // pointer to the mesh (level) object
  elem *el = msh->el;  // pointer to the elem object in msh (level)

  MultiLevelSolution *mlSol = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution *sol = ml_prob._ml_sol->GetSolutionLevel (level);   // pointer to the solution (level) object

  LinearEquationSolver *pdeSys = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  SparseMatrix *KK = pdeSys->_KK;  // pointer to the global stiffness matrix object in pdeSys (level)
  NumericVector *RES = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = 2;
  const unsigned  DIM = 3;
  unsigned iproc = msh->processor_id(); // get the process_id (for parallel computation)

  //solution variable
  unsigned solDxIndex[DIM];
  solDxIndex[0] = mlSol->GetIndex ("Dx1"); // get the position of "DX" in the ml_sol object
  solDxIndex[1] = mlSol->GetIndex ("Dx2"); // get the position of "DY" in the ml_sol object
  solDxIndex[2] = mlSol->GetIndex ("Dx3"); // get the position of "DZ" in the ml_sol object
  unsigned solType;
  solType = mlSol->GetSolutionType (solDxIndex[0]);  // get the finite element type for "U"
  std::vector < double > solx[DIM];  // surface coordinates
  std::vector < double > solDx[DIM];  // surface coordinates
  std::vector < double > solOldDx[DIM];  // surface coordinates
  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  unsigned solNDxIndex[DIM];
  solNDxIndex[0] = mlSol->GetIndex ("nDx1");   // get the position of "Y1" in the ml_sol object
  solNDxIndex[1] = mlSol->GetIndex ("nDx2");   // get the position of "Y2" in the ml_sol object
  solNDxIndex[2] = mlSol->GetIndex ("nDx3");   // get the position of "Y3" in the ml_sol object
  unsigned solNDxPdeIndex[DIM];
  solNDxPdeIndex[0] = mlPdeSys->GetSolPdeIndex ("nDx1");   // get the position of "Y1" in the pdeSys object
  solNDxPdeIndex[1] = mlPdeSys->GetSolPdeIndex ("nDx2");   // get the position of "Y2" in the pdeSys object
  solNDxPdeIndex[2] = mlPdeSys->GetSolPdeIndex ("nDx3");   // get the position of "Y3" in the pdeSys object
  std::vector < adept::adouble > solNDx[DIM]; // local Y solution


  unsigned solLIndex;
  solLIndex = mlSol->GetIndex ("Lambda1");   // get the position of "lambda" in the ml_sol object
  unsigned solLType;
  solLType = mlSol->GetSolutionType (solLIndex);  // get the finite element type for "lambda"
  unsigned solLPdeIndex;
  solLPdeIndex = mlPdeSys->GetSolPdeIndex ("Lambda1");   // get the position of "lambda" in the pdeSys object
  std::vector < adept::adouble > solL; // local lambda solution

  std::vector< int > SYSDOF; // local to global pdeSys dofs

  vector< double > Res; // local redidual vector
  std::vector< adept::adouble > aResNDx[3]; // local redidual vector
  std::vector< adept::adouble > aResL; // local redidual vector


  vector < double > Jac; // local Jacobian matrix (ordered by column, adept)

  KK->zero();  // Set to zero all the entries of the Global Matrix
  RES->zero(); // Set to zero all the entries of the Global Residual

  // element loop: each process loops only on the elements that owns
  for (int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

    short unsigned ielGeom = msh->GetElementType (iel);
    unsigned nxDofs  = msh->GetElementDofNumber (iel, solType);   // number of solution element dofs
    unsigned nLDofs  = msh->GetElementDofNumber (iel, solLType);   // number of solution element dofs


    for (unsigned K = 0; K < DIM; K++) {
      solDx[K].resize (nxDofs);
      solOldDx[K].resize (nxDofs);
      solx[K].resize (nxDofs);
      solNDx[K].resize (nxDofs);
      solL.resize (nLDofs);
    }

    // resize local arrays
    SYSDOF.resize (DIM * nxDofs + nLDofs);
    Res.resize (DIM * nxDofs + nLDofs);       //resize

    for (unsigned K = 0; K < DIM; K++) {
      aResNDx[K].assign (nxDofs, 0.);  //resize and zet to zero
    }
    aResL.assign (nLDofs, 0.);


    // local storage of global mapping and solution
    for (unsigned i = 0; i < nxDofs; i++) {
      unsigned iDDof = msh->GetSolutionDof (i, iel, solType); // global to local mapping between solution node and solution dof
      unsigned iXDof  = msh->GetSolutionDof (i, iel, xType);
      for (unsigned K = 0; K < DIM; K++) {
        solx[K][i] = (*msh->_topology->_Sol[K]) (iXDof) + (*sol->_Sol[solDxIndex[K]]) (iDDof);
        solDx[K][i] = (*sol->_Sol[solDxIndex[K]]) (iDDof);
        solOldDx[K][i] = (*sol->_SolOld[solDxIndex[K]]) (iDDof);
        solNDx[K][i] = (*sol->_Sol[solNDxIndex[K]]) (iDDof);
        SYSDOF[ K * nxDofs + i] = pdeSys->GetSystemDof (solNDxIndex[K], solNDxPdeIndex[K], i, iel); // global to global mapping between solution node and pdeSys dof
      }
    }

    // local storage of global mapping and solution
    for (unsigned i = 0; i < nLDofs; i++) {
      unsigned iLDof = msh->GetSolutionDof (i, iel, solLType); // global to local mapping between solution node and solution dof
      solL[i] = (*sol->_Sol[solLIndex]) (iLDof); // global to local solution
      SYSDOF[DIM * nxDofs + i] = pdeSys->GetSystemDof (solLIndex, solLPdeIndex, i, iel);  // global to global mapping between solution node and pdeSys dof
    }


    // start a new recording of all the operations involving adept::adouble variables
    s.new_recording();

    // *** Gauss point loop ***
    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solType]->GetGaussPointNumber(); ig++) {

      const double *phix;  // local test function
      const double *phix_uv[dim]; // local test function first order partial derivatives

      double weight; // gauss point weight

      // *** get gauss point weight, test function and test function partial derivatives ***
      phix = msh->_finiteElement[ielGeom][solType]->GetPhi (ig);
      phix_uv[0] = msh->_finiteElement[ielGeom][solType]->GetDPhiDXi (ig); //derivative in u
      phix_uv[1] = msh->_finiteElement[ielGeom][solType]->GetDPhiDEta (ig); //derivative in v

      weight = msh->_finiteElement[ielGeom][solType]->GetGaussWeight (ig);

      double solOldDxg[3] = {0., 0., 0.};
      double solDxg[3] = {0., 0., 0.};
      double solx_uv[3][2] = {{0., 0.}, {0., 0.}, {0., 0.}};

      adept::adouble solNDx_uv[3][2] = {{0., 0.}, {0., 0.}, {0., 0.}};
      double solDxOld_uv[3][2] = {{0., 0.}, {0., 0.}, {0., 0.}};

      adept::adouble solNDxg[3] = {0., 0., 0.};


      for (unsigned K = 0; K < DIM; K++) {
        for (unsigned i = 0; i < nxDofs; i++) {
          solOldDxg[K] += phix[i] * solOldDx[K][i];
          solDxg[K] += phix[i] * solDx[K][i];
          solNDxg[K] += phix[i] * solNDx[K][i];
        }
        for (int j = 0; j < dim; j++) {
          for (unsigned i = 0; i < nxDofs; i++) {
            solx_uv[K][j]     += phix_uv[j][i] * solx[K][i];
            solDxOld_uv[K][j] += phix_uv[j][i] * solOldDx[K][i];
            solNDx_uv[K][j]   += phix_uv[j][i] * solNDx[K][i];
          }
        }
      }


      double g[dim][dim] = {{0., 0.}, {0., 0.}};
      for (unsigned i = 0; i < dim; i++) {
        for (unsigned j = 0; j < dim; j++) {
          for (unsigned K = 0; K < DIM; K++) {
            g[i][j] += solx_uv[K][i] * solx_uv[K][j];
          }
        }
      }
      double detg = g[0][0] * g[1][1] - g[0][1] * g[1][0];

      double normal[DIM];
      normal[0] = (solx_uv[1][0] * solx_uv[2][1] - solx_uv[2][0] * solx_uv[1][1]) / sqrt (detg);
      normal[1] = (solx_uv[2][0] * solx_uv[0][1] - solx_uv[0][0] * solx_uv[2][1]) / sqrt (detg);
      normal[2] = (solx_uv[0][0] * solx_uv[1][1] - solx_uv[1][0] * solx_uv[0][1]) / sqrt (detg);

      adept::adouble DnXmDxdotN = 0.;
      for (unsigned K = 0; K < DIM; K++) {
        DnXmDxdotN += (solDxg[K] - solNDxg[K]) * normal[K];
      }

      double Area = weight * sqrt (detg);

      double gi[dim][dim];
      gi[0][0] =  g[1][1] / detg;
      gi[0][1] = -g[0][1] / detg;
      gi[1][0] = -g[1][0] / detg;
      gi[1][1] =  g[0][0] / detg;


      double Jir[2][3] = {{0., 0., 0.}, {0., 0., 0.}};
      for (unsigned i = 0; i < dim; i++) {
        for (unsigned J = 0; J < DIM; J++) {
          for (unsigned k = 0; k < dim; k++) {
            Jir[i][J] += gi[i][k] * solx_uv[J][k];
          }
        }
      }

      adept::adouble solNDx_Xtan[DIM][DIM] = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};
      adept::adouble solDxOld_Xtan[DIM][DIM] = {{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}};

      for (unsigned I = 0; I < DIM; I++) {
        for (unsigned J = 0; J < DIM; J++) {
          for (unsigned k = 0; k < dim; k++) {
            solNDx_Xtan[I][J] += solNDx_uv[I][k] * Jir[k][J];
            solDxOld_Xtan[I][J] += solDxOld_uv[I][k] * Jir[k][J];
          }
        }
      }

      std::vector < adept::adouble > phix_Xtan[DIM];

      for (unsigned J = 0; J < DIM; J++) {
        phix_Xtan[J].assign (nxDofs, 0.);
        for (unsigned inode  = 0; inode < nxDofs; inode++) {
          for (unsigned k = 0; k < dim; k++) {
            phix_Xtan[J][inode] += phix_uv[k][inode] * Jir[k][J];
          }
        }
      }

      for (unsigned K = 0; K < DIM; K++) {
        //Energy equation
        for (unsigned i = 0; i < nxDofs; i++) {
          adept::adouble term1 = 0.;
          for (unsigned J = 0; J < DIM; J++) {
            if (J != K) {
              term1 += (solNDx_Xtan[K][J] + solNDx_Xtan[J][K]) * phix_Xtan[J][i];
            }
            else {
              term1 += 0.5 * (solNDx_Xtan[K][J] + solNDx_Xtan[J][K]) * phix_Xtan[J][i];
            }
          }
          aResNDx[K][i] += (term1 + solL[0] * phix[i] * normal[K]) * Area;
        }
      }

      aResL[0] += DnXmDxdotN * Area;




    } // end gauss point loop

    //--------------------------------------------------------------------------------------------------------
    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store


    for (int K = 0; K < DIM; K++) {
      for (int i = 0; i < nxDofs; i++) {
        Res[ K * nxDofs + i] = -aResNDx[K][i].value();
      }
    }

    for (int i = 0; i < nLDofs; i++) {
      Res[DIM * nxDofs + i] = - aResL[i].value();
    }

    RES->add_vector_blocked (Res, SYSDOF);



    Jac.resize ( (DIM * nxDofs + nLDofs) * (DIM * nxDofs + nLDofs));

    // define the dependent variables

    for (int K = 0; K < DIM; K++) {
      s.dependent (&aResNDx[K][0], nxDofs);
    }
    s.dependent (&aResL[0], nLDofs);


    // define the dependent variables

    for (int K = 0; K < DIM; K++) {
      s.independent (&solNDx[K][0], nxDofs);
    }
    s.independent (&solL[0], nLDofs);

    // get the jacobian matrix (ordered by row)
    s.jacobian (&Jac[0], true);

    KK->add_matrix_blocked (Jac, SYSDOF, SYSDOF);

    s.clear_independents();
    s.clear_dependents();

  } //end element loop for each process

  RES->close();
  KK->close();

  // ***************** END ASSEMBLY *******************
}
