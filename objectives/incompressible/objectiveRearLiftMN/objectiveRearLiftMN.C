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

#include "objectiveRearLiftMN.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

namespace objectives
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(objectiveRearLiftMN, 0);
addToRunTimeSelectionTable
(
    objectiveIncompressible,
    objectiveRearLiftMN,
    dictionary
);


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

objectiveRearLiftMN::objectiveRearLiftMN
(
    const fvMesh& mesh,
    const dictionary& dict,
    const word& adjointSolverName,
    const word& primalSolverName
)
:
    objectiveIncompressible(mesh, dict, adjointSolverName, primalSolverName),
    objectivePatches_
    (
        mesh_.boundaryMesh().patchSet
        (
            dict.get<wordRes>("patches")
        ).sortedToc()
    ),
    forceDirection_(dict.get<vector>("liftDir")),
    momentDirection_(dict.get<vector>("axis")),
    rotationCentre_(dict.get<vector>("rotationCenter")),
    Aref_(dict.get<scalar>("Aref")),
    lRef_(dict.get<scalar>("lRef")),
    UInf_(dict.get<scalar>("UInf")),
    invDenom_(2./(UInf_*UInf_*Aref_*lRef_)),
    invMomDenom_(2./(UInf_*UInf_*Aref_*lRef_)),
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
    if (objectivePatches_.empty())
    {
        FatalErrorInFunction
            << "No valid patch name on which to minimize " << type() << endl
            << exit(FatalError);
    }
    if (debug)
    {
        Info<< "Minimizing " << type() << " in patches:" << endl;
        for (const label patchI : objectivePatches_)
        {
            Info<< "\t " << mesh_.boundary()[patchI].name() << endl;
        }
    }

    // Allocate boundary field pointers
    bdJdpPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
    bdSdbMultPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
    bdxdbMultPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
    bdJdStressPtr_.reset(createZeroBoundaryPtr<tensor>(mesh_));
    bdxdbDirectMultPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

scalar objectiveRearLiftMN::J()
{
    vector pressureForce(Zero);
    vector viscousForce(Zero);
    vector cumulativeForce(Zero);

    vector pressureMoment(Zero);
    vector viscousMoment(Zero);
    vector cumulativeMoment(Zero);

    // Update field here and use the same value for all functions
    const volScalarField& p = vars_.pInst();
    devReff_ = vars_.turbulence()->devReff()();

    for (const label patchI : objectivePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        const vectorField& Sf = patch.Sf();
        vectorField dx(patch.Cf() - rotationCentre_);
        pressureMoment += gSum
        (
            (dx ^ Sf)*p.boundaryField()[patchI]
        );

        // Viscous term calculated using the full tensor derivative
        viscousMoment += gSum
        (
            (dx^(devReff_.boundaryField()[patchI] & Sf))
        );

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

    cumulativeMoment = pressureMoment + viscousMoment;
    cumulativeForce = pressureForce + viscousForce;

    scalar moment = cumulativeMoment & momentDirection_;
    scalar Cm = moment*invMomDenom_;

    scalar force = cumulativeForce & forceDirection_;
    scalar Cforce = force*invDenom_;

    scalar rearLift = force - (moment/lRef_);
    scalar CLr = rearLift * invDenom_;

    DebugInfo<<
        "Moment|Coeff " << moment << "|" << Cm << endl <<
        "Force|Coeff " << force << "|" << Cforce << endl <<
        "Rear Lift|Coeff " << rearLift << "|" << CLr << endl;

    J_ = CLr;
    return CLr;
}


void objectiveRearLiftMN::update_meanValues()
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


void objectiveRearLiftMN::update_boundarydJdp()
{
    for (const label patchI : objectivePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        vectorField dx(patch.Cf() - rotationCentre_);
        bdJdpPtr_()[patchI] = forceDirection_*invDenom_ - (momentDirection_ ^ dx)*invMomDenom_;
    }
}


void objectiveRearLiftMN::update_dSdbMultiplier()
{
    const volScalarField& p = vars_.p();

    for (const label patchI : objectivePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        const vectorField dx(patch.Cf() - rotationCentre_);
        bdSdbMultPtr_()[patchI] =
        (
            (
                forceDirection_& devReff_.boundaryField()[patchI]
            )
          + (forceDirection_)*p.boundaryField()[patchI]
        )
        *invDenom_
        -
        (
            (
                (
                    (momentDirection_^dx) &
                    (
                        devReff_.boundaryField()[patchI]
                    )
                )
            )
          + (momentDirection_^dx) * p.boundaryField()[patchI]
        )
        *invMomDenom_;
    }
}


void objectiveRearLiftMN::update_dxdbMultiplier()
{
    const volScalarField& p = vars_.p();
    const volVectorField& U = vars_.U();

    const autoPtr<incompressible::RASModelVariables>&
       turbVars = vars_.RASModelVariables();
    const singlePhaseTransportModel& lamTransp = vars_.laminarTransport();

    volScalarField nuEff(lamTransp.nu() + turbVars->nutRef());
    volTensorField gradU(fvc::grad(U));
    volTensorField::Boundary& gradUbf = gradU.boundaryFieldRef();

    // Explicitly correct the boundary gradient to get rid of the
    // tangential component
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

    volVectorField gradp(fvc::grad(p));

    for (const label patchI : objectivePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        tmp<vectorField> tnf = patch.nf();
        const vectorField& nf = tnf();
        vectorField dx(patch.Cf() - rotationCentre_);
        vectorField aux(momentDirection_^dx);
        bdxdbMultPtr_()[patchI] =
        (
            (
                (
                   -(forceDirection_.x() * gradStressX.boundaryField()[patchI])
                   -(forceDirection_.y() * gradStressY.boundaryField()[patchI])
                   -(forceDirection_.z() * gradStressZ.boundaryField()[patchI])
                ) & nf
            )
            + (forceDirection_ & nf)*gradp.boundaryField()[patchI]
        )
        *invDenom_
        -
        (
            (
                (
                   -(aux.component(0) * gradStressX.boundaryField()[patchI])
                   -(aux.component(1) * gradStressY.boundaryField()[patchI])
                   -(aux.component(2) * gradStressZ.boundaryField()[patchI])
                ) & nf
            )
            + (momentDirection_ & (dx^nf))*gradp.boundaryField()[patchI]
        )
        *invMomDenom_;
    }
}


void objectiveRearLiftMN::update_dJdStressMultiplier()
{
    for (const label patchI : objectivePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        tmp<vectorField> tnf = patch.nf();
        const vectorField& nf = tnf();
        bdJdStressPtr_()[patchI] = (forceDirection_ * nf)*invDenom_;
    }
}


void objectiveRearLiftMN::update_dxdbDirectMultiplier()
{
    const volScalarField& p = vars_.p();

    for (const label patchI : objectivePatches_)
    {
        const fvPatch& patch = mesh_.boundary()[patchI];
        tmp<vectorField> tnf = patch.nf();
        const vectorField& nf = tnf();
        const vectorField force
        (
            (
                ((p.boundaryField()[patchI]*nf)
              + (devReff_.boundaryField()[patchI] & nf))
            )
        );
        bdxdbDirectMultPtr_()[patchI] =
            -(force^momentDirection_)*invMomDenom_;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace objectives
} // End namespace Foam

// ************************************************************************* //
