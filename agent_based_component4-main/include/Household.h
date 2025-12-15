#ifndef HOUSEHOLD
#define HOUSEHOLD

#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cstdint>

#include "repast_hpc/AgentId.h"
#include "repast_hpc/SharedContext.h"
#include "repast_hpc/SharedDiscreteSpace.h"
#include "repast_hpc/Random.h"
#include "Location.h"

class Household {
private:
    repast::AgentId householdId;
    Location* assignedField;
    int maizeStorage;
    int age;
    int deathAge;

    double totalDebt;
    double reputation;
    
    int yearsInSettlement;
    int totalYearsInSettlement;
    int surplusYears;

    double attachment;
    double stability;

    double deficitRatio;
    double bufferRatio;

    double val_A_request;
    double val_A_trans;
    double val_E;

public:
    Household(repast::AgentId id, int a, int deathAge, int mStorage);
    ~Household();

    virtual repast::AgentId& getId() { return householdId; }
    virtual const repast::AgentId& getId() const { return householdId; }

    Location* getAssignedField() { return assignedField; }
    
    int splitMaizeStored(int percentage);
    bool checkMaize(int needs);
    bool death();
    bool fission(int minFissionAge, int maxFissionAge, double gen, double fProb);
    void nextYear(int needs);
    void chooseField(Location* Field);

    int  getMaizeStorage() const { return maizeStorage; }
    void setMaizeStorage(int m)  { maizeStorage = m; }
    void addMaize(int delta)     { maizeStorage += delta; }

    int  getAge() const { return age; }

    int  getYearsInSettlement() const { return yearsInSettlement; }
    int  getTotalYearsInSettlement() const { return totalYearsInSettlement; }
    int  getSurplusYears() const { return surplusYears; }

    void incrementYearsInSettlement();
    void resetSettlementClock();
    void recordYearOutcome(bool surplus);

    double getAttachment() const { return attachment; }
    void   setAttachment(double a) { attachment = a; }

    double getStability() const { return stability; }
    void   setStability(double c) { stability = c; }

    double getDeficitRatio() const { return deficitRatio; }
    void   setDeficitRatio(double d) { deficitRatio = d; }

    double getBufferRatio() const { return bufferRatio; }
    void   setBufferRatio(double b) { bufferRatio = b; }

    double getReputation() const { return reputation; }
    void   setReputation(double r) { reputation = r; }

    double getTotalDebt() const { return totalDebt; }
    void   setTotalDebt(double d) { totalDebt = d; }
    void   addDebt(double delta) { totalDebt += delta; }

    void   setRequest(double r) { val_A_request = r; }
    double getRequest() const   { return val_A_request; }

    void   setTrans(double t)   { val_A_trans = t; }
    double getTrans() const     { return val_A_trans; }

    void   setExposure(double e) { val_E = e; }
    double getExposure() const   { return val_E; }
};

#endif
