/** tutorial/Ex1
 * This example shows how to:
 * initialize a femus application;
 * define the multilevel-mesh object mlMsh;
 * read from the file ./input/square.neu the coarse-level mesh and associate it to mlMsh;
 * add in mlMsh uniform refined level-meshes;
 * define the multilevel-solution object mlSol associated to mlMsh;
 * add in mlSol different types of finite element solution variables;
 * initialize the solution varables;
 * define vtk and gmv writer objects associated to mlSol;
 * print vtk and gmv binary-format files in ./output directory.
 **/

#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "GMVWriter.hpp"
#include "LinearImplicitSystem.hpp"
#include "NumericVector.hpp"

using namespace femus;

bool SetBoundaryCondition(const std::vector < double >& x, const char solName[], double& value, const int faceName, const double time) {
  bool dirichlet = true; 
  value = 0.;

  if (faceName == 1)    dirichlet = false;

  if (faceName == 3)    dirichlet = false;

  return dirichlet;
}

double GetExactSolutionLaplace(const std::vector < double >& x) {
  double pi = acos(-1.);
  return -pi * pi * cos(pi * x[0]) * cos(pi * x[1]) - pi * pi * cos(pi * x[0]) * cos(pi * x[1]);
};


double InitialValueU(const std::vector < double >& x) {
  return 0;
}

double InitialValueZ(const std::vector < double >& x) {
  return 0;
}

double InitialValueY(const std::vector < double >& x) {
  return 0;
}



void AssemblePoissonProblem(MultiLevelProblem& ml_prob);


int main(int argc, char** args) {

 //************* INITIALIZATION BEGIN **************************************************************************  
 // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);
 //************* INITIALIZATION END **************************************************************************  

//************* MESH BEGIN **************************************************************************  
  // define multilevel mesh
  MultiLevelMesh mlMsh;
 
  
  double scalingFactor = 1.;
  // read coarse level mesh and generate finers level meshes
  mlMsh.GenerateCoarseBoxMesh(4,4,0,-0.5,0.5,-0.5,0.5,0.,0.,QUAD9,"seventh");
  
  /* "seventh" is the order of accuracy that is used in the gauss integration scheme
      probably in the furure it is not going to be an argument of this function   */
  unsigned numberOfUniformLevels = 1;
  unsigned numberOfSelectiveLevels = 0;
  mlMsh.RefineMesh(numberOfUniformLevels , numberOfUniformLevels + numberOfSelectiveLevels, NULL);
  mlMsh.PrintInfo();
//************* MESH END **************************************************************************  

//************* SOLUTION BEGIN **************************************************************************  
  // define the multilevel solution and attach the mlMsh object to it
  MultiLevelSolution mlSol(&mlMsh);

  // add variables to mlSol
  mlSol.AddSolution("U", LAGRANGE, FIRST);
  mlSol.AddSolution("Z", LAGRANGE, FIRST);
  mlSol.AddSolution("Y", LAGRANGE, FIRST);

  mlSol.Initialize("All");    // initialize all varaibles to zero
  
  mlSol.Initialize("U",InitialValueU);
  mlSol.Initialize("Y",InitialValueY);
  mlSol.Initialize("Z",InitialValueZ);
  
  

       // attach the boundary condition function and generate boundary data
      mlSol.AttachSetBoundaryConditionFunction(SetBoundaryCondition);
      mlSol.GenerateBdc("U");
      mlSol.GenerateBdc("Y");
      mlSol.GenerateBdc("Z");
      
      
      
//************* SOLUTION END **************************************************************************  

//************* PROBLEM BEGIN **************************************************************************  
      // define the multilevel problem attach the mlSol object to it
      MultiLevelProblem mlProb(&mlSol);

      // add system Poisson in mlProb as a Linear Implicit System
      LinearImplicitSystem& system = mlProb.add_system < LinearImplicitSystem > ("Poisson");

      // add solution "U" to system
      system.AddSolutionToSystemPDE("U");
      system.AddSolutionToSystemPDE("Y");
      system.AddSolutionToSystemPDE("Z");
      
      // attach the assembling function to system
      system.SetAssembleFunction(AssemblePoissonProblem);

      // initilaize and solve the system
      system.init();
      system.solve();
//************* PROBLEM END **************************************************************************  
 
  
//************* PRINT BEGIN **************************************************************************  
  // print solutions
  std::vector < std::string > variablesToBePrinted;
  variablesToBePrinted.push_back("U");
  variablesToBePrinted.push_back("Y");
  variablesToBePrinted.push_back("Z");

  VTKWriter vtkIO(&mlSol);
  vtkIO.write(DEFAULT_OUTPUTDIR, "biquadratic", variablesToBePrinted);
//************* PRINT END **************************************************************************  

   return 0;
 }


void AssemblePoissonProblem(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data

  //  level is the level of the PDE system to be assembled
  //  levelMax is the Maximum level of the MultiLevelProblem
  //  assembleMatrix is a flag that tells if only the residual or also the matrix should be assembled

  //  extract pointers to the several objects that we are going to use

  LinearImplicitSystem* mlPdeSys  = &ml_prob.get_system<LinearImplicitSystem> ("Poisson");   // pointer to the linear implicit system named "Poisson"
  const unsigned level = mlPdeSys->GetLevelToAssemble();
  const unsigned levelMax = mlPdeSys->GetLevelMax();
  const bool assembleMatrix = mlPdeSys->GetAssembleMatrix();

  Mesh*                    msh = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object
  elem*                     el = msh->el;  // pointer to the elem object in msh (level)

  MultiLevelSolution*    mlSol = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*                sol = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  SparseMatrix*             KK = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*           RES = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  unsigned dim2 = (3 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27

  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)

// =========== GEOMETRY ===============    
  vector < vector < double > > x(dim);    // local coordinates
  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  for (unsigned i = 0; i < dim; i++) {
    x[i].reserve(maxSize);
  }
// =========== GEOMETRY ===============    

// =========== SOLUTION =============== 

// =============== U ==================
// ====================================
  vector <double> phi_U;  // local test function
  vector <double> phi_x_U; // local test function first order partial derivatives
  vector <double> phi_xx_U; // local test function second order partial derivatives
  phi_U.reserve(maxSize);
  phi_x_U.reserve(maxSize * dim);
  phi_xx_U.reserve(maxSize * dim2);
  
  unsigned soluIndexU;
  soluIndexU = mlSol->GetIndex("U");    // get the position of "U" in the ml_sol object
  unsigned soluTypeU = mlSol->GetSolutionType(soluIndexU);    // get the finite element type for "U"

  unsigned soluPdeIndexU;
  soluPdeIndexU = mlPdeSys->GetSolPdeIndex("U");    // get the position of "U" in the pdeSys object

  vector < double >  soluU; // local solution
  soluU.reserve(maxSize);
  
  vector< int > l2GMap_U; // local to global mapping
  l2GMap_U.reserve(maxSize);
//=====================================
//=====================================
  
// =============== Y ==================
// ==================================== 
  vector <double> phi_Y;  // local test function
  vector <double> phi_x_Y; // local test function first order partial derivatives
  vector <double> phi_xx_Y; // local test function second order partial derivatives
  phi_Y.reserve(maxSize);
  phi_x_Y.reserve(maxSize * dim);
  phi_xx_Y.reserve(maxSize * dim2);
  
  unsigned soluIndexY;
  soluIndexY = mlSol->GetIndex("Y");    // get the position of "Y" in the ml_sol object
  unsigned soluTypeY = mlSol->GetSolutionType(soluIndexY);    // get the finite element type for "Y"

  unsigned soluPdeIndexY;
  soluPdeIndexY = mlPdeSys->GetSolPdeIndex("Y");    // get the position of "Y" in the pdeSys object

  vector < double >  soluY; // local solution
  soluY.reserve(maxSize);
  
  vector< int > l2GMap_Y; // local to global mapping
  l2GMap_Y.reserve(maxSize);
//=====================================
//=====================================

// =============== Z ==================
// ==================================== 
  vector <double> phi_Z;  // local test function
  vector <double> phi_x_Z; // local test function first order partial derivatives
  vector <double> phi_xx_Z; // local test function second order partial derivatives
  phi_Z.reserve(maxSize);
  phi_x_Z.reserve(maxSize * dim);
  phi_xx_Z.reserve(maxSize * dim2);
  
  unsigned soluIndexZ;
  soluIndexZ = mlSol->GetIndex("Z");    // get the position of "Z" in the ml_sol object
  unsigned soluTypeZ = mlSol->GetSolutionType(soluIndexZ);    // get the finite element type for "Z"

  unsigned soluPdeIndexZ;
  soluPdeIndexZ = mlPdeSys->GetSolPdeIndex("Z");    // get the position of "Z" in the pdeSys object

  vector < double >  soluZ; // local solution
  soluZ.reserve(maxSize);
  
  vector< int > l2GMap_Z; // local to global mapping
  l2GMap_Z.reserve(maxSize);
//=====================================
//=====================================
  
  
// =========== SOLUTION ===============    


// =========== EQUATION ===============    
  double weight; // gauss point weight
  
  const int solType_max = 2;  //biquadratic

  const int n_vars = 3;
 
  vector< int > l2GMap_AllVars; // local to global mapping
  l2GMap_AllVars.reserve(n_vars*maxSize);

  vector< double > Res; // local redidual vector
  Res.reserve(maxSize);

  vector < double > Jac;
  Jac.reserve(maxSize * maxSize);
  
  if (assembleMatrix) { KK->zero(); }// Set to zero all the entries of the Global Matrix
   RES->zero();
  // =========== EQUATION ===============    

  // element loop: each process loops only on the elements that owns
  for (int iel = msh->IS_Mts2Gmt_elem_offset[iproc]; iel < msh->IS_Mts2Gmt_elem_offset[iproc + 1]; iel++) {

    unsigned kel = msh->IS_Mts2Gmt_elem[iel]; // mapping between paralell dof and mesh dof

// =========== GEOMETRY ===============    
    short unsigned kelGeom = el->GetElementType(kel);         // element geometry type
    unsigned nDofx = el->GetElementDofNumber(kel, xType);    // number of coordinate element dofs
    
    //resize
    for (int i = 0; i < dim; i++) {
      x[i].resize(nDofx);
    }
   
    //fill
   for (unsigned i = 0; i < nDofx; i++) {
      unsigned iNode = el->GetMeshDof(kel, i, xType);    // local to global coordinates node
      unsigned xDof  = msh->GetMetisDof(iNode, xType);    // global to global mapping between coordinates node and coordinate dof

      for (unsigned jdim = 0; jdim < dim; jdim++) {
        x[jdim][i] = (*msh->_coordinate->_Sol[jdim])(xDof);      // global extraction and local storage for the element coordinates
      }
    }
// =========== GEOMETRY ===============    
   
   
    
// =========== U ===============    
    unsigned nDofU  = el->GetElementDofNumber(kel, soluTypeU);

    //resize
    l2GMap_U.resize(nDofU);
    soluU.resize(nDofU);
    
    //fill
   for (unsigned i = 0; i < soluU.size(); i++) {
      unsigned iNode = el->GetMeshDof(kel, i, soluTypeU);
      unsigned solDofU = msh->GetMetisDof(iNode, soluTypeU);
      soluU[i] = (*sol->_Sol[soluIndexU])(solDofU);
      l2GMap_U[i] = pdeSys->GetKKDof(soluIndexU, soluPdeIndexU, iNode);
    }
// =========== U =============== 
 
 
// =========== Y =============== 
     unsigned nDofY  = el->GetElementDofNumber(kel, soluTypeY); 

    //resize
    l2GMap_Y.resize(nDofY);
    soluY.resize(nDofY);
    
    //fill
   for (unsigned i = 0; i < soluY.size(); i++) {
      unsigned iNode = el->GetMeshDof(kel, i, soluTypeY);   
      unsigned solDofY = msh->GetMetisDof(iNode, soluTypeY);   
      soluY[i] = (*sol->_Sol[soluIndexY])(solDofY);
      l2GMap_Y[i] = pdeSys->GetKKDof(soluIndexY, soluPdeIndexY, iNode);  
   }
// =========== Y ===============
      
// =========== Z =============== 
     unsigned nDofZ  = el->GetElementDofNumber(kel, soluTypeZ); 

    //resize
    l2GMap_Z.resize(nDofZ);
    soluZ.resize(nDofZ);
    
    //fill
   for (unsigned i = 0; i < soluZ.size(); i++) {
      unsigned iNode = el->GetMeshDof(kel, i, soluTypeZ); 
      unsigned solDofZ = msh->GetMetisDof(iNode, soluTypeZ);
      soluZ[i] = (*sol->_Sol[soluIndexZ])(solDofZ); 
      l2GMap_Z[i] = pdeSys->GetKKDof(soluIndexZ, soluPdeIndexZ, iNode); 
   }
// =========== Z ===============

    unsigned nDof_AllVars = nDofU + nDofY + nDofZ; 
    const int nDof_max    = nDofZ; 

 // =========== EQUATION ===============    
    Res.resize(nDof_AllVars);    //resize
    std::fill(Res.begin(), Res.end(), 0.);

    Jac.resize(nDof_AllVars * nDof_AllVars);    //resize
    std::fill(Jac.begin(), Jac.end(), 0.);
    
    l2GMap_AllVars.resize(0);
    l2GMap_AllVars.insert(l2GMap_AllVars.end(),l2GMap_U.begin(),l2GMap_U.end());
    l2GMap_AllVars.insert(l2GMap_AllVars.end(),l2GMap_Y.begin(),l2GMap_Y.end());
    l2GMap_AllVars.insert(l2GMap_AllVars.end(),l2GMap_Z.begin(),l2GMap_Z.end());
 // =========== EQUATION ===============    
    
   


    if (level == levelMax || !el->GetRefinedElementIndex(kel)) {      // do not care about this if now (it is used for the AMR)

      // *** Gauss point loop ***
      for (unsigned ig = 0; ig < msh->_finiteElement[kelGeom][solType_max]->GetGaussPointNumber(); ig++) {
        // *** get gauss point weight, test function and test function partial derivatives ***
        msh->_finiteElement[kelGeom][soluTypeU]->Jacobian(x, ig, weight, phi_U, phi_x_U, phi_xx_U);
	msh->_finiteElement[kelGeom][soluTypeY]->Jacobian(x, ig, weight, phi_Y, phi_x_Y, phi_xx_Y);
	msh->_finiteElement[kelGeom][soluTypeZ]->Jacobian(x, ig, weight, phi_Z, phi_x_Z, phi_xx_Z);

        // evaluate the solution, the solution derivatives and the coordinates in the gauss point
//         double solu_gss = 0;
//         vector < double > gradSolu_gss(dim, 0.);
//         vector < double > x_gss(dim, 0.);
// 
//         for (unsigned i = 0; i < nDofu; i++) {
//           solu_gss += phi[i] * solu[i];
// 
//           for (unsigned jdim = 0; jdim < dim; jdim++) {
//             gradSolu_gss[jdim] += phi_x[i * dim + jdim] * solu[i];
//             x_gss[jdim] += x[jdim][i] * phi[i];
//           }
//         }

        // *** phi_i loop ***
        for (unsigned i = 0; i < nDof_max; i++) {

          double srcTerm = 10.;
	  
          // FIRST ROW
	  if (i < nDofU)   Res[0    + 0      + i] += weight * srcTerm * phi_U[i] ;
          // SECOND ROW
          if (i < nDofY)   Res[nDofU + 0     + i] += weight * srcTerm * phi_Y[i] ;
          // THIRD ROW
          if (i < nDofZ)   Res[nDofU + nDofY + i] += weight * srcTerm * phi_Z[i] ;

          if (assembleMatrix) {
            // *** phi_j loop ***
            for (unsigned j = 0; j < nDof_max; j++) {
              double laplace_mat_U = 0.;
              double laplace_mat_Y = 0.;
              double laplace_mat_Z = 0.;

              for (unsigned kdim = 0; kdim < dim; kdim++) {
		if ( i < nDofU && j < nDofU )         laplace_mat_U        += (phi_x_U[i * dim + kdim] * phi_x_U[j * dim + kdim]);
		if ( i < nDofY && j < nDofY )         laplace_mat_Y        += (phi_x_Y[i * dim + kdim] * phi_x_Y[j * dim + kdim]);
		if ( i < nDofZ && j < nDofZ )         laplace_mat_Z        += (phi_x_Z[i * dim + kdim] * phi_x_Z[j * dim + kdim]);
		
              }

                if ( i < nDofU && j < nDofU )         Jac[               0 * (nDofU + nDofY + nDofZ) + i * (nDofU + nDofY + nDofZ) + (0              + j)]  += weight * laplace_mat_U;
		if ( i < nDofY && j < nDofY )         Jac[ (nDofU + 0)     * (nDofU + nDofY + nDofZ) + i * (nDofU + nDofY + nDofZ) + (nDofU          + j)]  += weight * laplace_mat_Y;
		if ( i < nDofZ && j < nDofZ )         Jac[ (nDofU + nDofY) * (nDofU + nDofY + nDofZ) + i * (nDofU + nDofY + nDofZ) + (nDofU  + nDofY + j)]  += weight * laplace_mat_Z;
 		
            } // end phi_j loop
          } // endif assemble_matrix
  
        } // end phi_i loop
      } // end gauss point loop
    } // endif single element not refined or fine grid loop

    //--------------------------------------------------------------------------------------------------------
    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store
    RES->add_vector_blocked(Res, l2GMap_AllVars);

    if (assembleMatrix) {
      //store K in the global matrix KK
      KK->add_matrix_blocked(Jac, l2GMap_AllVars, l2GMap_AllVars);
    }
  } //end element loop for each process

  RES->close();

  if (assembleMatrix) KK->close();

  // ***************** END ASSEMBLY *******************
}
