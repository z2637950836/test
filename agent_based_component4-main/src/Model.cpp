#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream> 
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cstdint>

#include <boost/mpi.hpp>
#include "repast_hpc/AgentId.h"
#include "repast_hpc/RepastProcess.h"
#include "repast_hpc/Utilities.h"
#include "repast_hpc/Properties.h"
#include "repast_hpc/initialize_random.h"
#include "repast_hpc/SVDataSetBuilder.h"
#include "repast_hpc/Point.h"
#include "repast_hpc/Random.h"
#include "repast_hpc/Schedule.h"
#include "repast_hpc/SharedContext.h"
#include "repast_hpc/SharedDiscreteSpace.h"
#include "repast_hpc/GridComponents.h"
#include "repast_hpc/Moore2DGridQuery.h"

#include "Model.h"
#include "Household.h" 

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

static void ensureOutputsDirExists() {
#ifdef _WIN32
    _mkdir("outputs");
#else
    mkdir("outputs", 0755);
#endif
}

AnasaziModel::AnasaziModel(std::string propsFile, int argc, char** argv, boost::mpi::communicator* comm): context(comm) , locationContext(comm) {
    props = new repast::Properties(propsFile, argc, argv, comm);
    infl.windowL = std::stoi(props->getProperty("influence.windowL"));
    infl.rho = std::stod(props->getProperty("influence.rho"));
    infl.k_d = std::stod(props->getProperty("influence.kd"));
    infl.k_b = std::stod(props->getProperty("influence.kb"));
    infl.k_a = std::stod(props->getProperty("influence.ka"));
    infl.k_c = std::stod(props->getProperty("influence.kc"));
    infl.k_e = std::stod(props->getProperty("influence.ke"));
    infl.probBase = std::stod(props->getProperty("influence.probability.base"));
    infl.capPerYear = std::stod(props->getProperty("influence.cap.per.year"));
    settle.lambda = std::stod(props->getProperty("influence.settle.lambda"));
    settle.alpha  = std::stod(props->getProperty("influence.settle.alpha"));
    settle.a      = std::stod(props->getProperty("influence.settle.a"));
    settle.b      = std::stod(props->getProperty("influence.settle.b"));
    social.ks     = std::stod(props->getProperty("social.ks"));
    social.k_up   = std::stod(props->getProperty("social.k_up"));
    social.k_down = std::stod(props->getProperty("social.k_down"));
    boardSizeX = repast::strToInt(props->getProperty("board.size.x"));
    boardSizeY = repast::strToInt(props->getProperty("board.size.y"));
    initializeRandom(*props, comm);
    repast::Point<double> origin(0,0);
    repast::Point<double> extent(boardSizeX, boardSizeY);
    repast::GridDimensions gd (origin, extent);
    int procX = repast::strToInt(props->getProperty("proc.per.x"));
    int procY = repast::strToInt(props->getProperty("proc.per.y"));
    int bufferSize = repast::strToInt(props->getProperty("grid.buffer"));
    std::vector<int> processDims; processDims.push_back(procX); processDims.push_back(procY);
    householdSpace = new repast::SharedDiscreteSpace<Household, repast::StrictBorders, repast::SimpleAdder<Household> >("AgentDiscreteSpace",gd,processDims,bufferSize, comm);
    locationSpace = new repast::SharedDiscreteSpace<Location, repast::StrictBorders, repast::SimpleAdder<Location> >("LocationDiscreteSpace",gd,processDims,bufferSize, comm);
    context.addProjection(householdSpace); locationContext.addProjection(locationSpace);
    param.startYear = repast::strToInt(props->getProperty("start.year"));
    param.endYear = repast::strToInt(props->getProperty("end.year"));
    param.maxStorageYear = repast::strToInt(props->getProperty("max.store.year"));
    param.maxStorage = repast::strToInt(props->getProperty("max.storage"));
    param.householdNeed = repast::strToInt(props->getProperty("household.need"));
    param.minFissionAge = repast::strToInt(props->getProperty("min.fission.age"));
    param.maxFissionAge = repast::strToInt(props->getProperty("max.fission.age"));
    param.minDeathAge = repast::strToInt(props->getProperty("min.death.age"));
    param.maxDeathAge = repast::strToInt(props->getProperty("max.death.age"));
    param.maxDistance = repast::strToInt(props->getProperty("max.distance"));
    param.initMinCorn = repast::strToInt(props->getProperty("initial.min.corn"));
    param.initMaxCorn = repast::strToInt(props->getProperty("initial.max.corn"));
    param.annualVariance = repast::strToDouble(props->getProperty("annual.variance"));
    param.spatialVariance = repast::strToDouble(props->getProperty("spatial.variance"));
    param.fertilityProbability = repast::strToDouble(props->getProperty("fertility.prop"));
    param.harvestAdjustment = repast::strToDouble(props->getProperty("harvest.adj"));
    param.maizeStorageRatio = repast::strToDouble(props->getProperty("new.household.ini.maize"));
    year = param.startYear;
    stopAt = param.endYear - param.startYear + 1;
    fissionGen = new repast::DoubleUniformGenerator(repast::Random::instance()->createUniDoubleGenerator(0,1));
    deathAgeGen = new repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(param.minDeathAge,param.maxDeathAge));
    yieldGen = new repast::NormalGenerator(repast::Random::instance()->createNormalGenerator(0,param.annualVariance));
    soilGen = new repast::NormalGenerator(repast::Random::instance()->createNormalGenerator(0,param.spatialVariance));
    initAgeGen = new repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(0,param.minDeathAge));
    initMaizeGen = new repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(param.initMinCorn,param.initMaxCorn));
    string resultFile = props->getProperty("result.file");
    out.open(resultFile); out << "Year,Number-of-Households" << endl;
}
AnasaziModel::~AnasaziModel() { delete props; out.close(); }

void AnasaziModel::initAgents() {
    int rank = repast::RepastProcess::instance()->rank();
    if (rank == 0) {
        ensureOutputsDirExists();
        std::ofstream debugFile;
        debugFile.open("outputs/check_samples_rank0.csv", std::ios::out | std::ios::trunc);
        if (debugFile.is_open()) { debugFile << "year,hh_id,x,y,storage,t,A_req,A_trans,D,B,A,C,E,z,p\n"; debugFile.close(); }
    }
    int LocationID = 0;
    for(int i=0; i<boardSizeX; i++ ) {
        for(int j=0; j<boardSizeY; j++) {
            repast::AgentId id(LocationID, rank, 1);
            Location* agent = new Location(id, soilGen->next());
            locationContext.addAgent(agent);
            locationSpace->moveTo(id, repast::Point<int>(i, j));
            LocationID++;
        }
    }
    readCsvMap(); readCsvWater(); readCsvPdsi(); readCsvHydro();
    int noOfAgents  = repast::strToInt(props->getProperty("count.of.agents"));
    repast::IntUniformGenerator xGen = repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(0,boardSizeX-1));
    repast::IntUniformGenerator yGen = repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(0,boardSizeY-1));
    for(int i=0; i<noOfAgents; i++) {
        repast::AgentId id(houseID, rank, 2);
        Household* agent = new Household(id, initAgeGen->next(), deathAgeGen->next(), initMaizeGen->next());
        agent->setReputation(0.1); agent->setAttachment(0.1);
        context.addAgent(agent);
        bool houseNotFound = true;
        do {
            int x = xGen.next(); int y = yGen.next();
            repast::Point<int> locationRepast(x,y);
            Location* randomLocation = locationSpace->getObjectAt(locationRepast);
            if(randomLocation && randomLocation->getState() != 2) {
                householdSpace->moveTo(id, locationRepast);
                randomLocation->setState(1);
                houseNotFound = false;
            }
        } while(houseNotFound);
        houseID++;
    }
    updateLocationProperties();
    repast::SharedContext<Household>::const_iterator local_agents_iter = context.begin();
    while(local_agents_iter != context.end()) {
        Household* household = (&**local_agents_iter);
        if(household->death()) {
            repast::AgentId id = household->getId();
            local_agents_iter++;
            std::vector<int> houseIntLocation;
            householdSpace->getLocation(id, houseIntLocation);
            if(!houseIntLocation.empty()) {
                Location* householdLocation = locationSpace->getObjectAt(repast::Point<int>(houseIntLocation));
                if(householdLocation) householdLocation->setState(0);
            }
            context.removeAgent(id);
        } else {
            local_agents_iter++;
            fieldSearch(household);
        }
    }
}

void AnasaziModel::doPerTick() {
    if (repast::RepastProcess::instance()->rank() == 0) std::cout << "Year: " << year << " Agents: " << context.size() << std::endl;
    updateLocationProperties();
    writeOutputToFile();
    year++;
    updateHouseholdProperties();
}

std::uint64_t AnasaziModel::keyOf(const repast::AgentId& id) const {
    std::uint64_t a = (std::uint64_t)(std::uint32_t)id.id();
    std::uint64_t r = (std::uint64_t)(std::uint16_t)id.startingRank();
    std::uint64_t t = (std::uint64_t)(std::uint16_t)id.agentType();
    return (a << 32) ^ (r << 16) ^ t;
}
bool AnasaziModel::isKin(std::uint64_t a, std::uint64_t b) const {
    auto ita = parentOf.find(a); if (ita != parentOf.end() && ita->second == b) return true;
    auto itb = parentOf.find(b); if (itb != parentOf.end() && itb->second == a) return true;
    return false;
}
double AnasaziModel::clamp01(double x) const { if (x < 0.0) return 0.0; if (x > 1.0) return 1.0; return x; }
double AnasaziModel::tieStrength(Household* b, Household* l) const {
    if(!l || !b) return 0.0; 
    double K = isKin(keyOf(b->getId()), keyOf(l->getId())) ? 1.0 : 0.0;
    return clamp01((1.0 - social.ks) * l->getReputation() + social.ks * K);
}
double AnasaziModel::randn01() const {
    double u1 = repast::Random::instance()->nextDouble();
    double u2 = repast::Random::instance()->nextDouble();
    if (u1 < 1e-12) u1 = 1e-12;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}
double AnasaziModel::truncNormal(double mu, double sigma, double lo, double hi, int maxTry) const {
    if (hi <= lo) return lo;
    if (sigma <= 1e-12) return (mu < lo) ? lo : ((mu > hi) ? hi : mu);
    for (int k = 0; k < maxTry; ++k) {
        double v = mu + sigma * randn01();
        if (v >= lo && v <= hi) return v;
    }
    return (mu < lo) ? lo : ((mu > hi) ? hi : mu);
}

void AnasaziModel::processSocialRiskPooling(std::vector<Household*>& highDebtHouseholds) {
    highDebtHouseholds.clear();
    std::vector<Household*> agents;
    context.selectAgents(repast::SharedContext<Household>::LOCAL, context.size(), agents);
    if (agents.empty()) return;
    std::unordered_map<std::uint64_t, Household*> id2hh;
    std::unordered_map<std::uint64_t, double> aidReceived;
    for (auto* h : agents) {
        if(!h) continue;
        std::uint64_t k = keyOf(h->getId());
        id2hh[k] = h; aidReceived[k] = 0.0;
        h->setRequest(0.0); h->setTrans(0.0); h->setExposure(0.0);
    }
    const int needs = param.householdNeed;
    for (auto* borrower : agents) {
        if(!borrower) continue;
        int storage_i = borrower->getMaizeStorage();
        if (storage_i <= needs) continue;
        auto itDebtRow = debt_ij.find(keyOf(borrower->getId()));
        if (itDebtRow == debt_ij.end() || itDebtRow->second.empty()) continue;
        struct LenderDebt { Household* lender; std::uint64_t kj; double amt; double S; };
        std::vector<LenderDebt> lenders;
        for (auto& kv : itDebtRow->second) {
            if (kv.second <= 1e-9) continue;
            auto itL = id2hh.find(kv.first);
            if (itL != id2hh.end() && itL->second) lenders.push_back({ itL->second, kv.first, kv.second, tieStrength(borrower, itL->second) });
        }
        std::sort(lenders.begin(), lenders.end(), [](const LenderDebt& a, const LenderDebt& b) { return a.S > b.S; });
        int surplus = storage_i - needs;
        for (auto& ld : lenders) {
            if (surplus <= 0) break;
            int debtInt = (int)std::floor(ld.amt + 1e-9);
            if (debtInt <= 0) continue;
            int rep = (surplus < debtInt) ? surplus : debtInt;
            surplus -= rep;
            ld.lender->setMaizeStorage(ld.lender->getMaizeStorage() + rep);
            itDebtRow->second[ld.kj] -= (double)rep;
            if (itDebtRow->second[ld.kj] <= 1e-9) itDebtRow->second.erase(ld.kj);
        }
        borrower->setMaizeStorage(needs + surplus);
        if (itDebtRow->second.empty()) debt_ij.erase(itDebtRow);
    }
    for (auto* borrower : agents) {
        if(!borrower) continue;
        int storage_i = borrower->getMaizeStorage();
        if (storage_i >= needs) continue;
        double A_request = (double)(needs - storage_i);
        borrower->setRequest(A_request);
        double A_trans_i = 0.0;
        struct LenderInfo { Household* lender; double S; };
        std::vector<LenderInfo> candidates;
        for (auto* lender : agents) {
            if (lender && lender != borrower) candidates.push_back({ lender, tieStrength(borrower, lender) });
        }
        std::sort(candidates.begin(), candidates.end(), [](const LenderInfo& a, const LenderInfo& b) { return a.S > b.S; });
        int asked = 0;
        for (auto& cand : candidates) {
            if (A_request <= 1e-6 || asked >= 3) break;
            int storage_j = cand.lender->getMaizeStorage();
            int A_ij = 0;
            if (storage_j > 0) {
                double sharelimit = truncNormal(storage_j/3.0, storage_j/4.0, 0.0, (double)storage_j);
                double give = std::min(A_request, sharelimit);
                A_ij = (int)std::floor(give + 1e-9);
                if (A_ij > 0) {
                    cand.lender->setMaizeStorage(storage_j - A_ij);
                    A_trans_i += (double)A_ij;
                    debt_ij[keyOf(borrower->getId())][keyOf(cand.lender->getId())] += (double)A_ij;
                    double f = (((double)A_ij/storage_j) + ((double)A_ij/A_request)) / 2.0;
                    cand.lender->setReputation(clamp01(cand.lender->getReputation() + social.k_up * f));
                } else cand.lender->setReputation(clamp01(cand.lender->getReputation() - social.k_down));
            }
            A_request -= (double)A_ij;
            asked++;
        }
        borrower->setMaizeStorage(storage_i + (int)std::floor(A_trans_i + 1e-9));
        borrower->setTrans(A_trans_i);
        aidReceived[keyOf(borrower->getId())] = A_trans_i;
        double debt_total = 0.0;
        auto itRow = debt_ij.find(keyOf(borrower->getId()));
        if (itRow != debt_ij.end()) for (auto& kv : itRow->second) debt_total += kv.second;
        if (debt_total >= (double)needs) highDebtHouseholds.push_back(borrower);
    }
    for (auto* h : agents) {
        if(!h) continue;
        int expHarvest = (h->getAssignedField()) ? h->getAssignedField()->getExpectedYield() : 0;
        double A_trans = aidReceived[keyOf(h->getId())];
        if (expHarvest < needs) {
            double D = ((double)needs - ((double)expHarvest + A_trans)) / (double)needs;
            if (D < 0.0) D = 0.0;
            double B = (needs - expHarvest > 1e-9) ? clamp01(A_trans / (needs - expHarvest)) : 0.0;
            h->setDeficitRatio(std::max(0.0, D)); h->setBufferRatio(B);
        } else { h->setDeficitRatio(0.0); h->setBufferRatio(0.0); }
    }
}

double AnasaziModel::computeExposure(Household* i, const std::vector<MigrantRecord>& natural_migrants) {
    if (natural_migrants.empty()) return 0.0;
    std::vector<int> loc_i; householdSpace->getLocation(i->getId(), loc_i);
    if (loc_i.size() < 2) return 0.0;
    double E = 0.0;
    for (const auto& rec : natural_migrants) {
        if (rec.id == i->getId()) continue;
        if (rec.location.empty() || rec.location.size() < 2) continue;
        double d = std::sqrt(std::pow(loc_i[0]-rec.location[0],2) + std::pow(loc_i[1]-rec.location[1],2));
        E += std::exp(-infl.rho * d);
    }
    return E;
}

void AnasaziModel::processSocialInfluenceMigration(const std::vector<MigrantRecord>& natural_migrants_records, std::vector<repast::AgentId>& followers_t) {
    std::vector<Household*> all; context.selectAgents(repast::SharedContext<Household>::LOCAL, context.size(), all);
    for (auto h : all) {
        if(!h) continue;
        double E = (!natural_migrants_records.empty()) ? computeExposure(h, natural_migrants_records) : 0.0;
        h->setExposure(E);
        if (E > 1e-9) {
            double z = infl.k_d * h->getDeficitRatio() - infl.k_b * h->getBufferRatio() - infl.k_a * h->getAttachment() - infl.k_c * h->getStability() + infl.k_e * E;
            if ((1.0 / (1.0 + std::exp(-z))) > infl.probBase) followers_t.push_back(h->getId());
        }
    }
    int cap = static_cast<int>(infl.capPerYear * all.size());
    if (cap < static_cast<int>(followers_t.size())) followers_t.resize(cap);
}

void AnasaziModel::logCheckSummary(int n, int f, int h, int s) {
    ensureOutputsDirExists();
    if (repast::RepastProcess::instance()->rank() != 0) return;
    std::string path = "outputs/check_summary_rank0.csv";
    static std::ofstream out; static bool opened = false;
    if (!opened) { out.open(path, std::ios::app); if (out.tellp()==0) out << "year,natural,followers,highDebt,survivors\n"; opened = true; }
    out << year << "," << n << "," << f << "," << h << "," << s << "\n";
}

void AnasaziModel::logCheckSamples(const std::vector<Household*>& survivors) {
    if (survivors.empty() || repast::RepastProcess::instance()->rank() != 0) return;
    ensureOutputsDirExists();
    std::string path = "outputs/check_samples_rank0.csv";
    static std::ofstream out; static bool opened = false;
    if (!opened) { out.open(path, std::ios::app); opened = true; }
    std::vector<Household*> borrowers; borrowers.reserve(survivors.size());
    for (auto* h : survivors) if (h && h->getRequest() > 0.001) borrowers.push_back(h);
    std::cout << "Year " << year << " Check: Found " << borrowers.size() << " active borrowers out of " << survivors.size() << " agents." << std::endl;
    
    // SAFEGUARD: Prevent crash if < 2 agents
    if (borrowers.size() < 2) return; 

    int n = (int)borrowers.size();
    repast::IntUniformGenerator pick(repast::Random::instance()->createUniIntGenerator(0, n - 1));
    int idx1 = pick.next();
    int idx2 = pick.next();
    if (idx2 == idx1) idx2 = (idx1 + 1) % n;

    auto writeOne = [&](Household* h){
        if(!h) return;
        double D=h->getDeficitRatio(), B=h->getBufferRatio(), A=h->getAttachment(), C=h->getStability(), E=h->getExposure();
        double z = infl.k_d*D - infl.k_b*B - infl.k_a*A - infl.k_c*C + infl.k_e*E;
        double p = 1.0 / (1.0 + std::exp(-z));
        std::vector<int> loc; householdSpace->getLocation(h->getId(), loc);
        out << year << "," << h->getId().id() << "," << (loc.size()>=2?loc[0]:-1) << "," << (loc.size()>=2?loc[1]:-1) << "," 
            << h->getMaizeStorage() << "," << h->getYearsInSettlement() << "," << h->getRequest() << "," << h->getTrans() << ","
            << D << "," << B << "," << A << "," << C << "," << E << "," << z << "," << p << "\n";
    };
    writeOne(borrowers[idx1]); writeOne(borrowers[idx2]); out.flush();
}

void AnasaziModel::updateHouseholdProperties() {
    if (context.size() < 1) return;
    std::vector<repast::AgentId> followers, natural, highDebtIds;
    std::vector<MigrantRecord> migrants_records, current_natural;
    std::vector<Household*> high_debt_hh;
    processSocialRiskPooling(high_debt_hh);
    for (auto* h : high_debt_hh) if(h) highDebtIds.push_back(h->getId());
    std::sort(highDebtIds.begin(), highDebtIds.end(), [&](const repast::AgentId& a, const repast::AgentId& b) { return keyOf(a) < keyOf(b); });
    highDebtIds.erase(std::unique(highDebtIds.begin(), highDebtIds.end(), [&](const repast::AgentId& a, const repast::AgentId& b) { return keyOf(a) == keyOf(b); }), highDebtIds.end());
    for (const auto& hid : highDebtIds) {
        std::vector<int> loc; householdSpace->getLocation(hid, loc);
        if(!loc.empty()) migrants_records.push_back({hid, loc});
        removeHouseholdById(hid);
    }
    std::vector<Household*> agents; context.selectAgents(repast::SharedContext<Household>::LOCAL, context.size(), agents);
    auto it = agents.begin();
    while (it != agents.end()) {
        Household* h = *it;
        if (!context.getAgent(h->getId())) { ++it; continue; }
        if (h->death()) { removeHouseholdById(h->getId()); ++it; continue; }
        if (!h->checkMaize(param.householdNeed)) {
            if (!fieldSearch(h)) {
                natural.push_back(h->getId());
                std::vector<int> loc; householdSpace->getLocation(h->getId(), loc);
                if (!loc.empty()) {
                    MigrantRecord rec = {h->getId(), loc};
                    migrants_records.push_back(rec); current_natural.push_back(rec);
                }
                removeHouseholdById(h->getId()); ++it; continue;
            }
        }
        if (h->fission(param.minFissionAge, param.maxFissionAge, fissionGen->next(), param.fertilityProbability)) {
            repast::AgentId id(houseID++, repast::RepastProcess::instance()->rank(), 2);
            Household* n = new Household(id, 0, deathAgeGen->next(), h->splitMaizeStored(param.maizeStorageRatio));
            n->setReputation(0.1); n->setAttachment(0.1);
            context.addAgent(n); parentOf[keyOf(n->getId())] = keyOf(h->getId());
            std::vector<int> loc; householdSpace->getLocation(h->getId(), loc);
            householdSpace->moveTo(id, repast::Point<int>(loc[0], loc[1]));
            fieldSearch(n);
        }
        h->nextYear(param.householdNeed); ++it;
    }
    std::vector<Household*> survivors; context.selectAgents(repast::SharedContext<Household>::LOCAL, context.size(), survivors);
    for (auto* h : survivors) {
        if(!h) continue;
        h->incrementYearsInSettlement(); h->recordYearOutcome(h->getMaizeStorage() > 0);
        int ty = h->getTotalYearsInSettlement();
        double C = (ty + settle.a + settle.b > 0) ? (h->getSurplusYears() + settle.a)/(double)(ty + settle.a + settle.b) : 0.0;
        h->setStability(C);
        double s_sum = 0.0; for (auto* o : survivors) if (o && o != h) s_sum += tieStrength(h, o);
        h->setAttachment(clamp01((1.0 - std::exp(-settle.lambda * h->getYearsInSettlement())) + settle.alpha * s_sum));
    }
    processSocialInfluenceMigration(current_natural, followers);
    logCheckSamples(survivors);
    logCheckSummary(natural.size(), followers.size(), highDebtIds.size(), survivors.size());
    std::sort(followers.begin(), followers.end(), [&](const repast::AgentId& a, const repast::AgentId& b) { return keyOf(a) < keyOf(b); });
    followers.erase(std::unique(followers.begin(), followers.end(), [&](const repast::AgentId& a, const repast::AgentId& b) { return keyOf(a) == keyOf(b); }), followers.end());
    for (const auto& hid : followers) {
        std::vector<int> loc; householdSpace->getLocation(hid, loc);
        if(!loc.empty()) migrants_records.push_back({hid, loc});
        removeHouseholdById(hid);
    }
    migrantsHistory.push_back(migrants_records);
    if (migrantsHistory.size() > (size_t)infl.windowL) migrantsHistory.erase(migrantsHistory.begin());
}

bool AnasaziModel::fieldSearch(Household* household) {
    if(!household) return false;
    std::vector<int> houseLoc; householdSpace->getLocation(household->getId(), houseLoc);
    if(houseLoc.size() < 2) return false; // SAFEGUARD
    repast::Moore2DGridQuery<Location> query(locationSpace);
    std::vector<Location*> neighbors, checked;
    int maxR = std::max(std::max(std::abs(houseLoc[0]-boardSizeX), houseLoc[0]), std::max(std::abs(houseLoc[1]-boardSizeY), houseLoc[1]));
    int r = 1;
    while(r <= maxR) {
        query.query(houseLoc, r, false, neighbors);
        for (auto* loc : neighbors) {
            if (std::find(checked.begin(), checked.end(), loc) != checked.end()) continue;
            checked.push_back(loc);
            if (loc->getState() == 0 && loc->getExpectedYield() >= param.householdNeed) {
                household->chooseField(loc); return true;
            }
        }
        neighbors.clear(); r++;
    }
    return relocateHousehold(household);
}

void AnasaziModel::removeHousehold(Household* household) {
    if(!household) return;
    repast::AgentId id = household->getId();
    if (!context.getAgent(id)) return;
    std::vector<int> loc; householdSpace->getLocation(id, loc);
    if (!loc.empty()) {
        Location* l = locationSpace->getObjectAt(repast::Point<int>(loc));
        std::vector<Household*> hlist; householdSpace->getObjectsAt(repast::Point<int>(loc), hlist);
        if (l && hlist.size()==1) l->setState(0);
    }
    Location* af = household->getAssignedField();
    if (af) {
        std::vector<int> floc; locationSpace->getLocation(af->getId(), floc);
        if (!floc.empty()) { Location* l = locationSpace->getObjectAt(repast::Point<int>(floc)); if(l) l->setState(0); }
    }
    std::uint64_t k = keyOf(id);
    debt_ij.erase(k);
    for (auto it = debt_ij.begin(); it != debt_ij.end(); ) {
        it->second.erase(k); if (it->second.empty()) it = debt_ij.erase(it); else ++it;
    }
    parentOf.erase(k);
    for (auto it = parentOf.begin(); it != parentOf.end(); ) {
        if (it->second == k) it = parentOf.erase(it); else ++it;
    }
    context.removeAgent(id);
}

bool AnasaziModel::relocateHousehold(Household* household) {
    if(!household) return false;
    std::vector<int> hLoc; householdSpace->getLocation(household->getId(), hLoc);
    if(hLoc.size() < 2) return false; // SAFEGUARD
    std::vector<Location*> neighbors, best;
    repast::Moore2DGridQuery<Location> query(locationSpace);
    query.query(hLoc, param.maxDistance, false, neighbors);
    for (auto* l : neighbors) if (l->getState() == 0) best.push_back(l);
    
    // SAFEGUARD: No valid location found
    if (best.empty()) return false;
    
    // SAFEGUARD: Random index safety
    int idx = 0;
    if (best.size() > 1) {
        idx = repast::Random::instance()->createUniIntGenerator(0, best.size()-1).next();
    }
    Location* newLoc = best[idx];
    
    Location* oldLoc = locationSpace->getObjectAt(repast::Point<int>(hLoc));
    if (oldLoc) oldLoc->setState(0);
    
    std::vector<int> nLoc; locationSpace->getLocation(newLoc->getId(), nLoc);
    householdSpace->moveTo(household->getId(), repast::Point<int>(nLoc));
    newLoc->setState(1);
    household->resetSettlementClock();
    return true;
}

double AnasaziModel::distanceBetween(const Household* a, const Household* b) const {
    std::vector<int> la, lb; householdSpace->getLocation(a->getId(), la); householdSpace->getLocation(b->getId(), lb);
    if (la.empty() || lb.empty()) return 0.0;
    return std::sqrt(std::pow(la[0]-lb[0],2) + std::pow(la[1]-lb[1],2));
}
Household* AnasaziModel::getHouseholdById(const repast::AgentId& id) { return context.getAgent(id); }
void AnasaziModel::readCsvMap() {
    int x,y,z,mz; string zone, mZone, t; std::ifstream f("data/map.csv"); f.ignore(500,'\n');
    while(std::getline(f,t,',')) {
        x=stoi(t); std::getline(f,t,','); y=stoi(t); std::getline(f,t,','); std::getline(f,zone,','); std::getline(f,mZone,'\n');
        if(zone=="\"Empty\"") z=0; else if(zone=="\"Natural\"") z=1; else if(zone=="\"Kinbiko\"") z=2; else if(zone=="\"Uplands\"") z=3;
        else if(zone=="\"North\"") z=4; else if(zone=="\"General\"") z=5; else if(zone=="\"North Dunes\"") z=6; else if(zone=="\"Mid Dunes\"") z=7;
        else if(zone=="\"Mid\"") z=8; else z=99;
        if(mZone.find("Empty")!=std::string::npos) mz=0; else if(mZone.find("No_Yield")!=std::string::npos) mz=1;
        else if(mZone.find("Yield_1")!=std::string::npos) mz=2; else if(mZone.find("Yield_2")!=std::string::npos) mz=3;
        else if(mZone.find("Yield_3")!=std::string::npos) mz=4; else if(mZone.find("Sand_dune")!=std::string::npos) mz=5; else mz=99;
        std::vector<Location*> l; locationSpace->getObjectsAt(repast::Point<int>(x,y), l); 
        // SAFEGUARD: Ensure l is not empty
        if(!l.empty()) l[0]->setZones(z,mz);
    }
}
void AnasaziModel::readCsvWater() {
    int tp,s,e,x,y; string t; std::ifstream f("data/water.csv"); f.ignore(500,'\n');
    while(std::getline(f,t,',')) {
        std::getline(f,t,','); std::getline(f,t,','); std::getline(f,t,','); tp=stoi(t); std::getline(f,t,','); s=stoi(t); std::getline(f,t,','); e=stoi(t);
        std::getline(f,t,','); x=stoi(t); std::getline(f,t,'\n'); y=stoi(t);
        std::vector<Location*> l; locationSpace->getObjectsAt(repast::Point<int>(x,y), l);
        // SAFEGUARD: Ensure l is not empty
        if(!l.empty()) l[0]->addWaterSource(tp,s,e);
    }
}
void AnasaziModel::readCsvPdsi() {
    int i=0; string t; std::ifstream f("data/pdsi.csv"); f.ignore(500,'\n');
    while(std::getline(f,t,',')) {
        // BOUNDS CHECK: Prevent array overflow
        if(i >= 2000) break; 
        pdsi[i].year=stoi(t); std::getline(f,t,','); pdsi[i].pdsiGeneral=stod(t); std::getline(f,t,','); pdsi[i].pdsiNorth=stod(t);
        std::getline(f,t,','); pdsi[i].pdsiMid=stod(t); std::getline(f,t,','); pdsi[i].pdsiNatural=stod(t); std::getline(f,t,','); pdsi[i].pdsiUpland=stod(t);
        std::getline(f,t,'\n'); pdsi[i].pdsiKinbiko=stod(t); i++;
    }
}
void AnasaziModel::readCsvHydro() {
    int i=0; string t; std::ifstream f("data/hydro.csv"); f.ignore(500,'\n');
    while(std::getline(f,t,',')) {
        // BOUNDS CHECK: Prevent array overflow
        if(i >= 2000) break;
        hydro[i].year=stoi(t); std::getline(f,t,','); hydro[i].hydroGeneral=stod(t); std::getline(f,t,','); hydro[i].hydroNorth=stod(t);
        std::getline(f,t,','); hydro[i].hydroMid=stod(t); std::getline(f,t,','); hydro[i].hydroNatural=stod(t); std::getline(f,t,','); hydro[i].hydroUpland=stod(t);
        std::getline(f,t,'\n'); hydro[i].hydroKinbiko=stod(t); i++;
    }
}
int AnasaziModel::yieldFromPdsi(int z, int mz) {
    int v, r, c;
    // BOUNDS CHECK: Index safety
    int idx = year - param.startYear;
    if (idx < 0 || idx >= 2000) return 0;

    switch(z){ case 1:v=pdsi[idx].pdsiNatural;break; case 2:v=pdsi[idx].pdsiKinbiko;break; case 3:v=pdsi[idx].pdsiUpland;break;
    case 4:case 6:v=pdsi[idx].pdsiNorth;break; case 5:v=pdsi[idx].pdsiGeneral;break; case 7:case 8:v=pdsi[idx].pdsiMid;break; default:return 0;}
    
    if(v<-3)r=0; else if(v<-1)r=1; else if(v<1)r=2; else if(v<3)r=3; else r=4;
    
    if(mz>=2)c=mz-2; else return 0;
    // BOUNDS CHECK for yieldLevels (assuming size 5x4 or similar)
    if(r < 0 || r >= 5 || c < 0 || c >= 4) return 0; 
    
    return yieldLevels[r][c];
}
double AnasaziModel::hydroLevel(int z) {
    int idx = year - param.startYear;
    if (idx < 0 || idx >= 2000) return 0;
    switch(z){ case 1:return hydro[idx].hydroNatural; case 2:return hydro[idx].hydroKinbiko; case 3:return hydro[idx].hydroUpland;
    case 4:case 6:return hydro[idx].hydroNorth; case 5:return hydro[idx].hydroGeneral; case 7:case 8:return hydro[idx].hydroMid; default:return 0;}
}
void AnasaziModel::checkWaterConditions() {
    existStreams = ((year>=280&&year<360)||(year>=800&&year<930)||(year>=1300&&year<1450));
    existAlluvium = ((year>=420&&year<560)||(year>=630&&year<680)||(year>=980&&year<1120)||(year>=1180&&year<1230));
}
void AnasaziModel::writeOutputToFile() { out << year << "," << context.size() << std::endl; }
void AnasaziModel::updateLocationProperties() {
    checkWaterConditions();
    for(int i=0; i<boardSizeX; i++) for(int j=0; j<boardSizeY; j++) {
        std::vector<Location*> l; locationSpace->getObjectsAt(repast::Point<int>(i,j), l);
        // CRITICAL FIX: Guard against empty vector and null pointer
        if(!l.empty() && l[0]) {
            l[0]->checkWater(existStreams,existAlluvium,i,j,year);
            l[0]->calculateYield(yieldFromPdsi(l[0]->getZone(),l[0]->getMaizeZone()), param.harvestAdjustment, yieldGen->next());
        }
    }
}
void AnasaziModel::initSchedule(repast::ScheduleRunner& runner) {
    runner.scheduleEvent(1, 1, repast::Schedule::FunctorPtr(new repast::MethodFunctor<AnasaziModel>(this, &AnasaziModel::doPerTick)));
    runner.scheduleStop(stopAt);
}
void AnasaziModel::removeHouseholdById(const repast::AgentId& id) {
    Household* h = context.getAgent(id);
    if (h) removeHousehold(h);
}
