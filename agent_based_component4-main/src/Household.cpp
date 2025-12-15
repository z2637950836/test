#include <boost/mpi.hpp>
#include <stdio.h>
#include <iostream>
#include "Household.h"
#include "repast_hpc/AgentId.h"
#include "repast_hpc/SharedContext.h"
#include "repast_hpc/SharedDiscreteSpace.h"
#include "repast_hpc/Random.h"

Household::Household(repast::AgentId id, int a, int deAge, int mStorage) {
    householdId = id;
    age = a;
    deathAge = deAge;
    maizeStorage = mStorage;
    assignedField = NULL;

    reputation = 0.5;
    totalDebt  = 0.0;

    yearsInSettlement      = 0;
    totalYearsInSettlement = 0;
    surplusYears           = 0;

    attachment   = 0.0;
    stability    = 0.0;
    deficitRatio = 0.0;
    bufferRatio  = 0.0;
    
    val_A_request = 0.0;
    val_A_trans = 0.0;
    val_E = 0.0;
}

Household::~Household() {
}

int Household::splitMaizeStored(int percentage) {
    int maizeEndowment = 0;
    double ratio = (double)percentage / 100.0;
    // Handle confusing input (e.g. 0.33 vs 33)
    if (percentage <= 1 && percentage > 0) ratio = (double)percentage;

    maizeEndowment = (int)(maizeStorage * ratio);
    maizeStorage = maizeStorage - maizeEndowment;
    return maizeEndowment;
}

bool Household::checkMaize(int needs) {
    if(assignedField == NULL) return false;
    
    if((assignedField->getExpectedYield() + maizeStorage) >= needs) {
        return true;
    } else {
        return false;
    }
}

bool Household::death() {
    return (age >= deathAge);
}

bool Household::fission(int minFissionAge, int maxFissionAge, double gen, double fProb) {
    return ((age >= minFissionAge && age <= maxFissionAge) && (gen <= fProb));
}

void Household::nextYear(int needs) {
    age++;
    int yield = 0;
    
    // SAFEGUARD: Check for null pointer
    if (assignedField != NULL) {
        yield = assignedField->getExpectedYield();
    }
    
    maizeStorage = yield + maizeStorage - needs;
}

void Household::chooseField(Location* Field) {
    if (assignedField != NULL) {
        assignedField->setState(0);
    }

    if (Field != NULL) {
        Field->setState(2);
    }

    assignedField = Field;
}

void Household::incrementYearsInSettlement() {
    yearsInSettlement++;
}

void Household::resetSettlementClock() {
    yearsInSettlement      = 0;
    totalYearsInSettlement = 0;
    surplusYears           = 0;
}

void Household::recordYearOutcome(bool surplus) {
    totalYearsInSettlement++;
    if (surplus) surplusYears++;
}
