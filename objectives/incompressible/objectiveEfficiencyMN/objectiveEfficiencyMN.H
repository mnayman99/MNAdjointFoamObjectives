/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2007-2020 PCOpt/NTUA
    Copyright (C) 2013-2020 FOSS GP
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.


Class
    Foam::objectives::objectiveEfficiencyMN

Description

SourceFiles
    objectiveEfficiencyMN.C

\*---------------------------------------------------------------------------*/

#ifndef objectiveEfficiencyMN_H
#define objectiveEfficiencyMN_H

#include "objectiveIncompressible.H"
#include "wallFvPatch.H"
#include "createZeroField.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

namespace objectives
{

/*---------------------------------------------------------------------------*\
                        Class objectiveEfficiencyMN Declaration
\*---------------------------------------------------------------------------*/

class objectiveEfficiencyMN
:
    public objectiveIncompressible
{
protected:

    // Protected Data

        labelHashSet forcePatches_;
        vector liftDirection_;
        vector dragDirection_;

        autoPtr<volVectorField> stressXPtr_;
        autoPtr<volVectorField> stressYPtr_;
        autoPtr<volVectorField> stressZPtr_;

        // Store this in order to computed only once per objective call
        volSymmTensorField devReff_;


public:

    //- Runtime type information
    TypeName("efficiencyMN");


    // Constructors

        //- Construct from components
        objectiveEfficiencyMN
        (
            const fvMesh& mesh,
            const dictionary& dict,
            const word& adjointSolverName,
            const word& primalSolverName
        );


    //- Destructor
    virtual ~objectiveEfficiencyMN() = default;


    // Member Functions

        //- Return the objective function value
        scalar J();

        //- Update mean drag and lift values
        void update_meanValues();

        //- Update values to be added to the adjoint wall velocity
        void update_boundarydJdp();

        //- Update delta(n dS)/delta b multiplier
        void update_dSdbMultiplier();

        //- Update delta(x)/delta b multiplier
        void update_dxdbMultiplier();

        //- Update dJ/dStress multiplier
        void update_dJdStressMultiplier();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace objectives
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
