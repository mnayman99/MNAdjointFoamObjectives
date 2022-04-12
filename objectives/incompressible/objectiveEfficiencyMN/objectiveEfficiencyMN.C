/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2007-2019 PCOpt/NTUA
    Copyright (C) 2013-2019 FOSS GP
    Copyright (C) 2019-2020 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "objectiveEfficiencyMN.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

namespace objectives
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(objectiveEfficiencyMN, 0);
addToRunTimeSelectionTable
(
    objectiveIncompressible,
    objectiveEfficiencyMN,
    dictionary
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

objectiveEfficiencyMN::objectiveEfficiencyMN
(
    const fvMesh& mesh,
    const dictionary& dict,
    const word& adjointSolverName,
    const word& primalSolverName
)
:
    objectiveIncompressible(mesh, dict, adjointSolverName, primalSolverName),
    forcePatches_
    (
        mesh_.boundaryMesh().patchSet
        (
            dict.get<wordRes>("patches")
        ).sortedToc()
    ),
    liftDirection_(dict.get<vector>("liftDirection")),
    dragDirection_(dict.get<vector>("dragDirection")),
    stressXPtr_
    (
        Foam::createZeroFieldPtr<vector>
        (
            mesh_, "stressX", dimLength/sqr(dimTime)
        )
    ),
    stressYPtr_
    (
        Foam::createZeroFieldPtr<vector>
        (
            mesh_, "stressY", dimLength/sqr(dimTime)
        )
    ),
    stressZPtr_
    (
        Foam::createZeroFieldPtr<vector>
        (
            mesh_, "stressZ", dimLength/sqr(dimTime)
        )
    ),
    devReff_(vars_.turbulence()->devReff()())
{
    // Sanity check and print info
    if (forcePatches_.empty())
    {
        FatalErrorInFunction
            << "No valid patch name on which to minimize " << type() << endl
            << exit(FatalError);
    }
    if (debug)
    {
        Info<< "Minimizing " << type() << " in patches:" << endl;
        for (const label patchI : forcePatches_)
        {
            Info<< "\t " << mesh_.boundary()[patchI].name() << endl;
        }
    }

    // Allocate boundary field pointers
    bdJdpPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
    bdSdbMultPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
    bdxdbMultPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
    bdJdStressPtr_.reset(createZeroBoundaryPtr<tensor>(mesh_));
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

scalar objectiveEfficiencyMN::J()
{
    vector pressureForce(Zero);
    vector viscousForce(Zero);
    vector cumulativeForce(Zero);


    const volScalarField& p = vars_.pInst();
    devReff_ = vars_.turbulence()->devReff()();

    for (const label patchI : forcePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        const vectorField& Sf = patch.Sf();
        pressureForce += gSum
        (
            Sf * p.boundaryField()[patchI]
        );
        // Viscous term calculated using the full tensor derivative
        viscousForce += gSum
        (
            devReff_.boundaryField()[patchI] & Sf
        );
    }

    cumulativeForce = pressureForce + viscousForce;

    scalar efficiency = (cumulativeForce & liftDirection_) / (cumulativeForce & dragDirection_);

    DebugInfo
        << "Effieciency " << efficiency << endl;

    J_ = efficiency;

    return efficiency;
}


void objectiveEfficiencyMN::update_meanValues()
{
    if (computeMeanFields_)
    {
        const volVectorField& U = vars_.U();
        const autoPtr<incompressible::RASModelVariables>&
           turbVars = vars_.RASModelVariables();
        const singlePhaseTransportModel& lamTransp = vars_.laminarTransport();

        devReff_ = turbVars->devReff(lamTransp, U)();
    }
}


void objectiveEfficiencyMN::update_boundarydJdp()
{
    // Compute contributions with mean fields, if present
    const volScalarField& p = vars_.p();

    for (const label patchI : forcePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        tmp<vectorField> tnf = patch.nf();
        const vectorField& nf = tnf();
        
        // First calculate the total force coefficient vector on the specific patch, which will be decomposed into lift and drag
        // Total force is calculated using nf to get a face-centred value for sensitivity contributions
        // vectorFields and scalarFields used, as the results isn't a simple sum as in the objective function derivation
        const vectorField totalForce((nf * p.boundaryField()[patchI]) + (devReff_.boundaryField()[patchI] & nf));
        const scalarField liftForce(totalForce & liftDirection_);
        const scalarField dragForce(totalForce & dragDirection_);

        bdJdpPtr_()[patchI] = (dragForce * liftDirection_ - liftForce * dragDirection_)/(dragForce * dragForce);
    }
}


void objectiveEfficiencyMN::update_dSdbMultiplier()
{
    // Compute contributions with mean fields, if present
    const volScalarField& p = vars_.p();
    
    for (const label patchI : forcePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        tmp<vectorField> tnf = patch.nf();
        const vectorField& nf = tnf();

        // First calculate the total force coefficient vector on the specific patch, which will be decomposed into lift and drag
        // Total force is calculated using nf to get a face-centred value for sensitivity contributions
        // vectorFields and scalarFields used, as the results isn't a simple sum as in the objective function derivation
        const vectorField totalForce((nf * p.boundaryField()[patchI]) + (devReff_.boundaryField()[patchI] & nf));
        const scalarField liftForce(totalForce & liftDirection_);
        const scalarField dragForce(totalForce & dragDirection_);

        vectorField liftContribution((liftDirection_& devReff_.boundaryField()[patchI]) + (liftDirection_)*p.boundaryField()[patchI]);
        vectorField dragContribution((dragDirection_& devReff_.boundaryField()[patchI]) + (dragDirection_)*p.boundaryField()[patchI]);

        bdSdbMultPtr_()[patchI].component(0) = liftContribution.component(0)/dragContribution.component(0);
        bdSdbMultPtr_()[patchI].component(1) = liftContribution.component(1)/dragContribution.component(1);
        bdSdbMultPtr_()[patchI].component(2) = liftContribution.component(2)/dragContribution.component(2);
    }
}


void objectiveEfficiencyMN::update_dxdbMultiplier()
{
    const volScalarField& p = vars_.p();
    const volVectorField& U = vars_.U();

    const autoPtr<incompressible::RASModelVariables>&
        turbVars = vars_.RASModelVariables();
    const singlePhaseTransportModel& lamTransp = vars_.laminarTransport();

    volScalarField nuEff(lamTransp.nu() + turbVars->nutRef());
    volTensorField gradU(fvc::grad(U));
    volTensorField::Boundary& gradUbf = gradU.boundaryFieldRef();

    // Explicitly correct the boundary gradient to get rid of
    // the tangential component
    forAll(mesh_.boundary(), patchI)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        if (isA<wallFvPatch>(patch))
        {
            tmp<vectorField> nf = patch.nf();
            gradUbf[patchI] = nf*U.boundaryField()[patchI].snGrad();
        }
    }

    volTensorField stress(nuEff*(gradU + T(gradU)));

    stressXPtr_().replace(0, stress.component(0));
    stressXPtr_().replace(1, stress.component(1));
    stressXPtr_().replace(2, stress.component(2));

    stressYPtr_().replace(0, stress.component(3));
    stressYPtr_().replace(1, stress.component(4));
    stressYPtr_().replace(2, stress.component(5));

    stressZPtr_().replace(0, stress.component(6));
    stressZPtr_().replace(1, stress.component(7));
    stressZPtr_().replace(2, stress.component(8));

    volTensorField gradStressX(fvc::grad(stressXPtr_()));
    volTensorField gradStressY(fvc::grad(stressYPtr_()));
    volTensorField gradStressZ(fvc::grad(stressZPtr_()));

    // the notorious second-order derivative at the wall. Use with caution!
    volVectorField gradp(fvc::grad(p));

    for (const label patchI : forcePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        tmp<vectorField> tnf = patch.nf();
        const vectorField& nf = tnf();
        
        // First calculate the total force coefficient vector on the specific patch, which will be decomposed into lift and drag
        // Total force is calculated using nf to get a face-centred value for sensitivity contributions
        // vectorFields and scalarFields used, as the results isn't a simple sum as in the objective function derivation
        const vectorField totalForce((nf * p.boundaryField()[patchI]) + (devReff_.boundaryField()[patchI] & nf));
        const scalarField liftForce(totalForce & liftDirection_);
        const scalarField dragForce(totalForce & dragDirection_);

        vectorField liftContribution = 
        (
            (
                (
                   -(liftDirection_.x() * gradStressX.boundaryField()[patchI])
                   -(liftDirection_.y() * gradStressY.boundaryField()[patchI])
                   -(liftDirection_.z() * gradStressZ.boundaryField()[patchI])
                ) & nf
            )
            + (liftDirection_ & nf)*gradp.boundaryField()[patchI]
        );

        vectorField dragContribution = 
        (
            (
                (
                   -(dragDirection_.x() * gradStressX.boundaryField()[patchI])
                   -(dragDirection_.y() * gradStressY.boundaryField()[patchI])
                   -(dragDirection_.z() * gradStressZ.boundaryField()[patchI])
                ) & nf
            )
            + (dragDirection_ & nf)*gradp.boundaryField()[patchI]
        );

        bdxdbMultPtr_()[patchI] =
        (
            (dragForce * liftContribution - liftForce * dragContribution)
            /
            (dragForce * dragForce)
        );
    }
}


void objectiveEfficiencyMN::update_dJdStressMultiplier()
{
    // Compute contributions with mean fields, if present
    const volScalarField& p = vars_.p();

    for (const label patchI : forcePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        tmp<vectorField> tnf = patch.nf();
        const vectorField& nf = tnf();
        
        // First calculate the total force coefficient vector on the specific patch, which will be decomposed into lift and drag
        // Total force is calculated using nf to get a face-centred value for sensitivity contributions
        // vectorFields and scalarFields used, as the results isn't a simple sum as in the objective function derivation
        const vectorField totalForce((nf * p.boundaryField()[patchI]) + (devReff_.boundaryField()[patchI] & nf));
        const scalarField liftForce(totalForce & liftDirection_);
        const scalarField dragForce(totalForce & dragDirection_);

        bdJdStressPtr_()[patchI] = (((liftDirection_ * nf) * dragForce) + ((dragDirection_ * nf) * liftForce)) / (dragForce * dragForce);
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace objectives
} // End namespace Foam

// ************************************************************************* //
