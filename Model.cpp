#include <stdio.h>
#include <vector>
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
#include <string>
#include <fstream>
#include <stdlib.h>
#include "repast_hpc/Moore2DGridQuery.h"
#include <cmath>
#include <initializer_list>
#include <unordered_map> 
#include <algorithm> 
#include <cstdint>
#include "Model.h"


#include <fstream>
#include <iomanip>
#include <string>

#ifdef _WIN32
#include <direct.h>   // _mkdir
#else
#include <sys/stat.h> // mkdir
#include <sys/types.h>
#endif



// substracts b<T> to a<T>
template <typename T>
void substract_vector(std::vector<T>& a, const std::vector<T>& b)
{
	typename std::vector<T>::iterator       it = a.begin();
	typename std::vector<T>::const_iterator it2 = b.begin();

	while (it != a.end())
	{
		while (it2 != b.end() && it != a.end())
		{
			if (*it == *it2)
			{
				it = a.erase(it);
				it2 = b.begin();
			}

			else
				++it2;
		}
		if (it != a.end())
			++it;

		it2 = b.begin();
	}
}

AnasaziModel::AnasaziModel(std::string propsFile, int argc, char** argv, boost::mpi::communicator* comm) : context(comm), locationContext(comm)
{


	props = new repast::Properties(propsFile, argc, argv, comm);
	//add new parameters
	infl.windowL = std::stoi(props->getProperty("influence.windowL"));
	infl.rho = std::stod(props->getProperty("influence.rho"));

	infl.k_d = std::stod(props->getProperty("influence.kd"));
	infl.k_b = std::stod(props->getProperty("influence.kb"));
	infl.k_a = std::stod(props->getProperty("influence.ka"));
	infl.k_c = std::stod(props->getProperty("influence.kc"));
	infl.k_e = std::stod(props->getProperty("influence.ke"));

	infl.probBase = std::stod(props->getProperty("influence.probability.base"));
	infl.capPerYear = std::stod(props->getProperty("influence.cap.per.year"));

	// -------------------- Feature 2: settlement attachment params --------------------
	settle.lambda = std::stod(props->getProperty("influence.settle.lambda"));
	settle.alpha = std::stod(props->getProperty("influence.settle.alpha"));
	settle.a = std::stod(props->getProperty("influence.settle.a"));
	settle.b = std::stod(props->getProperty("influence.settle.b"));
	// -------------------- Feature 1: social risk-pooling / reputation params --------------------
	social.ks = std::stod(props->getProperty("social.ks"));
	social.k_up = std::stod(props->getProperty("social.k_up"));
	social.k_down = std::stod(props->getProperty("social.k_down"));




	boardSizeX = repast::strToInt(props->getProperty("board.size.x"));
	boardSizeY = repast::strToInt(props->getProperty("board.size.y"));

	initializeRandom(*props, comm);
	repast::Point<double> origin(0, 0);
	repast::Point<double> extent(boardSizeX, boardSizeY);
	repast::GridDimensions gd(origin, extent);

	int procX = repast::strToInt(props->getProperty("proc.per.x"));
	int procY = repast::strToInt(props->getProperty("proc.per.y"));
	int bufferSize = repast::strToInt(props->getProperty("grid.buffer"));

	std::vector<int> processDims;
	processDims.push_back(procX);
	processDims.push_back(procY);
	householdSpace = new repast::SharedDiscreteSpace<Household, repast::StrictBorders, repast::SimpleAdder<Household> >("AgentDiscreteSpace", gd, processDims, bufferSize, comm);
	locationSpace = new repast::SharedDiscreteSpace<Location, repast::StrictBorders, repast::SimpleAdder<Location> >("LocationDiscreteSpace", gd, processDims, bufferSize, comm);

	context.addProjection(householdSpace);
	locationContext.addProjection(locationSpace);

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
	fissionGen = new repast::DoubleUniformGenerator(repast::Random::instance()->createUniDoubleGenerator(0, 1));
	deathAgeGen = new repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(param.minDeathAge, param.maxDeathAge));
	yieldGen = new repast::NormalGenerator(repast::Random::instance()->createNormalGenerator(0, param.annualVariance));
	soilGen = new repast::NormalGenerator(repast::Random::instance()->createNormalGenerator(0, param.spatialVariance));
	initAgeGen = new repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(0, param.minDeathAge));
	initMaizeGen = new repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(param.initMinCorn, param.initMaxCorn));

	string resultFile = props->getProperty("result.file");
	out.open(resultFile);
	out << "Year,Number-of-Households" << endl;
}

AnasaziModel::~AnasaziModel()
{
	delete props;
	out.close();
}

void AnasaziModel::initAgents()
{
	int rank = repast::RepastProcess::instance()->rank();

	int LocationID = 0;
	for (int i = 0; i < boardSizeX; i++)
	{
		for (int j = 0; j < boardSizeY; j++)
		{
			repast::AgentId id(LocationID, rank, 1);
			Location* agent = new Location(id, soilGen->next());
			locationContext.addAgent(agent);
			locationSpace->moveTo(id, repast::Point<int>(i, j));
			LocationID++;
		}
	}

	readCsvMap();
	readCsvWater();
	readCsvPdsi();
	readCsvHydro();
	int noOfAgents = repast::strToInt(props->getProperty("count.of.agents"));
	repast::IntUniformGenerator xGen = repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(0, boardSizeX - 1));
	repast::IntUniformGenerator yGen = repast::IntUniformGenerator(repast::Random::instance()->createUniIntGenerator(0, boardSizeY - 1));
	for (int i = 0; i < noOfAgents; i++)
	{
		repast::AgentId id(houseID, rank, 2);
		int initAge = initAgeGen->next();
		int mStorage = initMaizeGen->next();
		Household* agent = new Household(id, initAge, deathAgeGen->next(), mStorage);
		context.addAgent(agent);

		bool houseNotFound = true;
		do {
			int x = xGen.next();
			int y = yGen.next();
			repast::Point<int> locationRepast(x, y);
			Location* randomLocation = locationSpace->getObjectAt(locationRepast);
			if (randomLocation->getState() != 2) {  // as long is not a field (it could be empty or the residence of another household)
				householdSpace->moveTo(id, locationRepast);
				randomLocation->setState(1);
				houseNotFound = false;
			}
		} while (houseNotFound);

		houseID++;
	}

	updateLocationProperties();

	repast::SharedContext<Household>::const_iterator local_agents_iter = context.begin();
	repast::SharedContext<Household>::const_iterator local_agents_end = context.end();

	while (local_agents_iter != local_agents_end)
	{
		Household* household = (&**local_agents_iter);
		if (household->death())
		{
			repast::AgentId id = household->getId();
			local_agents_iter++;

			std::vector<int> houseIntLocation;
			householdSpace->getLocation(id, houseIntLocation);

			std::vector<Location*> locationList;
			if (!houseIntLocation.empty())
			{
				Location* householdLocation = locationSpace->getObjectAt(repast::Point<int>(houseIntLocation));
				householdLocation->setState(0);  // set the household residence to empty
			}
			context.removeAgent(id);
		}
		else
		{
			local_agents_iter++;
			fieldSearch(household);
		}
	}
}

void AnasaziModel::doPerTick()
{
	updateLocationProperties();
	writeOutputToFile();
	year++;
	updateHouseholdProperties();
}
//------------add new
std::uint64_t AnasaziModel::keyOf(const repast::AgentId& id) const {
	// 说明：repast::AgentId 常见接口是 id.id(), id.startingRank(), id.agentType()
	// 如果你那版 Repast 名字不同（如 currentRank），把这里改一下即可。
	std::uint64_t a = (std::uint64_t)(std::uint32_t)id.id();
	std::uint64_t r = (std::uint64_t)(std::uint16_t)id.startingRank();
	std::uint64_t t = (std::uint64_t)(std::uint16_t)id.agentType();
	return (a << 32) ^ (r << 16) ^ t;
}

bool AnasaziModel::isKin(std::uint64_t a, std::uint64_t b) const {
	auto ita = parentOf.find(a);
	if (ita != parentOf.end() && ita->second == b) return true; // a 的父是 b
	auto itb = parentOf.find(b);
	if (itb != parentOf.end() && itb->second == a) return true; // b 的父是 a
	return false;
}

double AnasaziModel::clamp01(double x) const {
	if (x < 0.0) return 0.0;
	if (x > 1.0) return 1.0;
	return x;
}

double AnasaziModel::tieStrength(Household* borrower, Household* lender) const {
	std::uint64_t ki = keyOf(borrower->getId());
	std::uint64_t kj = keyOf(lender->getId());
	double K = isKin(ki, kj) ? 1.0 : 0.0; // K_ij ∈ {0,1}
	// S_ij = clamp((1-k_s)*rep_j + k_s*K_ij, 0, 1)
	double S = (1.0 - social.ks) * lender->getReputation() + social.ks * K;
	return clamp01(S);
}

// Box–Muller: 用 repast::Random 的 nextDouble 造 N(0,1)
double AnasaziModel::randn01() const {
	double u1 = repast::Random::instance()->nextDouble();
	double u2 = repast::Random::instance()->nextDouble();
	if (u1 < 1e-12) u1 = 1e-12;
	return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// 截断正态 TN(mu, sigma^2, lo, hi)
double AnasaziModel::truncNormal(double mu, double sigma, double lo, double hi, int maxTry) const {
	if (hi <= lo) return lo;
	if (sigma <= 1e-12) {
		double v = mu;
		if (v < lo) v = lo;
		if (v > hi) v = hi;
		return v;
	}
	for (int k = 0; k < maxTry; ++k) {
		double v = mu + sigma * randn01();
		if (v >= lo && v <= hi) return v;
	}
	// 兜底：clamp（避免死循环）
	double v = mu;
	if (v < lo) v = lo;
	if (v > hi) v = hi;
	return v;
}

void AnasaziModel::processSocialRiskPooling(std::vector<Household*>& highDebtHouseholds)
{
	highDebtHouseholds.clear();

	// 当前存活家庭（本进程 LOCAL）
	std::vector<Household*> agents;
	context.selectAgents(repast::SharedContext<Household>::LOCAL, context.size(), agents);
	if (agents.empty()) return;

	// key -> Household*（用于过滤“债权人已经不在 context”的情况）
	std::unordered_map<std::uint64_t, Household*> id2hh;
	id2hh.reserve(agents.size() * 2);
	for (auto* h : agents) id2hh[keyOf(h->getId())] = h;

	// 记录本年每个 i 收到的净援助 A_trans_i（用于 D_i / B_i）
	std::unordered_map<std::uint64_t, double> aidReceived;
	aidReceived.reserve(agents.size() * 2);

	for (auto* h : agents) aidReceived[keyOf(h->getId())] = 0.0;
	//15 记录本年每个 i 的原始请求量 A_request（用于 B_i = A_trans_i / A_request）
	std::unordered_map<std::uint64_t, double> reqMade;
	reqMade.reserve(agents.size() * 2);
	for (auto* h : agents) reqMade[keyOf(h->getId())] = 0.0;


	const int needs = param.householdNeed;


	// =========================================================
	// (A) 债务偿还：若 maizeStorage_i > 0，则按 S_ij 从高到低还 Debt_ij
	// =========================================================
	for (auto* borrower : agents)
	{
		std::uint64_t ki = keyOf(borrower->getId());

		int storage_i = borrower->getMaizeStorage();
		if (storage_i <= 0) continue;

		auto itDebtRow = debt_ij.find(ki);
		if (itDebtRow == debt_ij.end() || itDebtRow->second.empty()) continue;

		struct LenderDebt {
			Household* lender;
			std::uint64_t kj;
			double amt;
			double S;
		};

		std::vector<LenderDebt> lenders;
		lenders.reserve(itDebtRow->second.size());

		for (auto& kv : itDebtRow->second)
		{
			std::uint64_t kj = kv.first;
			double amt = kv.second;
			if (amt <= 1e-9) continue;

			auto itL = id2hh.find(kj);
			if (itL == id2hh.end()) continue; // lender 已不在 context

			Household* lender = itL->second;
			double S = tieStrength(borrower, lender);
			lenders.push_back({ lender, kj, amt, S });
		}

		if (lenders.empty()) continue;

		std::sort(lenders.begin(), lenders.end(),
			[](const LenderDebt& a, const LenderDebt& b) { return a.S > b.S; });

		int available = storage_i;

		for (auto& ld : lenders)
		{
			if (available <= 0) break;

			int debtInt = (int)std::floor(ld.amt + 1e-9);
			if (debtInt <= 0) continue;

			int rep = (available < debtInt) ? available : debtInt;

			// borrower 出粮
			available -= rep;

			// lender 入粮
			int lenderStorage = ld.lender->getMaizeStorage();
			ld.lender->setMaizeStorage(lenderStorage + rep);

			// 更新债务
			itDebtRow->second[ld.kj] -= (double)rep;
			if (itDebtRow->second[ld.kj] <= 1e-9) {
				itDebtRow->second.erase(ld.kj);
			}
		}

		borrower->setMaizeStorage(available);

		// 清理空行（避免 debt_ij 长期堆垃圾）
		if (itDebtRow->second.empty()) {
			debt_ij.erase(itDebtRow);
		}
	}

	// =========================================================
	// (B) 借贷援助：若 maizeStorage_i < 0，则向 S_ij 最大的 lender 借，最多 3 个
	//    sharelimit_j ~ TN(mu, sigma^2, 0, maizeStorage_j)
	//    mu = maizeStorage_j/3, sigma = maizeStorage_j/4
	//    声誉更新：帮助 / （有余粮却拒绝）惩罚
	// =========================================================
	for (auto* borrower : agents)
	{
		std::uint64_t ki = keyOf(borrower->getId());

		int storage_i = borrower->getMaizeStorage();
		if (storage_i >= 0) continue;
		//15在 (B) 借贷开始处，把原始 A_request0 存进 reqMade

		double A_request0 = (double)(-storage_i);  // 原始请求量
		double A_request = A_request0;            // 下面循环里会被递减
		reqMade[ki] = A_request0;                  // 记下来给 (C) 算 B 用

		double A_trans_i = 0.0;


		struct LenderInfo { Household* lender; double S; };
		std::vector<LenderInfo> candidates;
		candidates.reserve(agents.size());

		for (auto* lender : agents) {
			if (lender == borrower) continue;
			double S = tieStrength(borrower, lender);
			candidates.push_back({ lender, S });
		}

		std::sort(candidates.begin(), candidates.end(),
			[](const LenderInfo& a, const LenderInfo& b) { return a.S > b.S; });

		int asked = 0;
		for (auto& cand : candidates)
		{
			if (A_request <= 1e-6) break;
			if (asked >= 3) break;

			Household* lender = cand.lender;
			std::uint64_t kj = keyOf(lender->getId());

			int storage_j_before = lender->getMaizeStorage();
			int A_ij = 0; // 默认 0：拒绝或无粮

			if (storage_j_before > 0)
			{
				double req_before = A_request; // 用于计算 f_j 的分母（更稳定）

				double mu = storage_j_before / 3.0;
				double sigma = storage_j_before / 4.0;

				double sharelimit = truncNormal(mu, sigma, 0.0, (double)storage_j_before);
				double give = std::min(req_before, sharelimit);

				A_ij = (int)std::floor(give + 1e-9);

				if (A_ij > 0)
				{
					lender->setMaizeStorage(storage_j_before - A_ij);
					A_trans_i += (double)A_ij;

					// Debt_ij += A_ij
					debt_ij[ki][kj] += (double)A_ij;

					// rep increase: f = ((A_ij/Mj) + (A_ij/A_req))/2
					double f = (((double)A_ij / (double)storage_j_before) + ((double)A_ij / req_before)) / 2.0;
					lender->setReputation(clamp01(lender->getReputation() + social.k_up * f));
				}
				else
				{
					// 15有余粮却拒绝
					if (storage_j_before > 0) {
						lender->setReputation(clamp01(lender->getReputation() - social.k_down));
					}
				}
			}
			else
			{
				// maizeStorage_j == 0：不惩罚
			}

			A_request -= (double)A_ij;
			asked++;
		}

		// borrower 获得净援助
		borrower->setMaizeStorage(storage_i + (int)std::floor(A_trans_i + 1e-9));
		aidReceived[ki] = A_trans_i;

		// debt_total >= needs -> 高债务迁出
		double debt_total = 0.0;
		auto itRow = debt_ij.find(ki);
		if (itRow != debt_ij.end()) {
			for (auto& kv : itRow->second) debt_total += kv.second;
		}
		if (debt_total >= (double)needs) {
			highDebtHouseholds.push_back(borrower);
		}
	}

	// =========================================================
	// (C) 计算 D_i / B_i（给 Feature 3 用）
	//     仅当 expectedHarvest < needs 时计算，否则 D=B=0
	// =========================================================
	for (auto* h : agents)
	{
		int expectedHarvest = 0;
		if (h->getAssignedField() != NULL) {
			expectedHarvest = h->getAssignedField()->getExpectedYield();
		}

		double A_trans_i = aidReceived[keyOf(h->getId())];
		double A_req_i = reqMade[keyOf(h->getId())];
		//15(C) 的 for 循环里（建议放在循环开头，拿到 A_req_i 之后）
		static int lastPrintYear = -1;
		static int printedThisYear = 0;
		if (lastPrintYear != year) {   // 新的一年，重置计数
			lastPrintYear = year;
			printedThisYear = 0;
		}
		int rank = repast::RepastProcess::instance()->rank();

		if (A_req_i > 1e-9) {
			int ms = h->getMaizeStorage();            // 借粮后最终的净粮
			double D = (ms < 0) ? (-ms / (double)needs) : 0.0;  // 缺口占 needs 的比例
			D = clamp01(D);

			//15B is social aid buffering
			double B = clamp01(A_trans_i / A_req_i);   // clamp到[0,1]
			//15在算出 B 之后加打印（只在 rank0 且每年最多 1 条
			if (rank == 0 && printedThisYear < 1) {
				printf("[year=%d] id=%d startRank=%d A_trans=%.3f A_req=%.3f B=%.3f\n",
					year, h->getId().id(), h->getId().startingRank(),
					A_trans_i, A_req_i, B);
				printedThisYear++;
			}

			h->setDeficitRatio(D);
			h->setBufferRatio(B);
		}
		else {
			h->setDeficitRatio(0.0);
			h->setBufferRatio(0.0);
		}

	}
}



void AnasaziModel::initSchedule(repast::ScheduleRunner& runner)
{
	runner.scheduleEvent(1, 1, repast::Schedule::FunctorPtr(new repast::MethodFunctor<AnasaziModel>(this, &AnasaziModel::doPerTick)));
	runner.scheduleStop(stopAt);
}

void AnasaziModel::readCsvMap()
{
	int x, y, z, mz;
	string zone, maizeZone, temp;

	std::ifstream file("data/map.csv");//define file object and open map.csv
	file.ignore(500, '\n');//Ignore first line

	while (1)//read until end of file
	{
		getline(file, temp, ',');
		if (!temp.empty())
		{
			x = repast::strToInt(temp); //Read until ',' and convert to int & store in x
			getline(file, temp, ',');
			y = repast::strToInt(temp); //Read until ',' and convert to int & store in y
			getline(file, temp, ','); //colour
			getline(file, zone, ',');// read until ',' and store into zone
			getline(file, maizeZone, '\n');// read until next line and store into maizeZone
			if (zone == "\"Empty\"")
			{
				z = 0;
			}
			else if (zone == "\"Natural\"")
			{
				z = 1;
			}
			else if (zone == "\"Kinbiko\"")
			{
				z = 2;
			}
			else if (zone == "\"Uplands\"")
			{
				z = 3;
			}
			else if (zone == "\"North\"")
			{
				z = 4;
			}
			else if (zone == "\"General\"")
			{
				z = 5;
			}
			else if (zone == "\"North Dunes\"")
			{
				z = 6;
			}
			else if (zone == "\"Mid Dunes\"")
			{
				z = 7;
			}
			else if (zone == "\"Mid\"")
			{
				z = 8;
			}
			else
			{
				z = 99;
			}

			if (maizeZone.find("Empty") != std::string::npos)
			{
				mz = 0;
			}
			else if (maizeZone.find("No_Yield") != std::string::npos)
			{
				mz = 1;
			}
			else if (maizeZone.find("Yield_1") != std::string::npos)
			{
				mz = 2;
			}
			else if (maizeZone.find("Yield_2") != std::string::npos)
			{
				mz = 3;
			}
			else if (maizeZone.find("Yield_3") != std::string::npos)
			{
				mz = 4;
			}
			else if (maizeZone.find("Sand_dune") != std::string::npos)
			{
				mz = 5;
			}
			else
			{
				mz = 99;
			}
			std::vector<Location*> locationList;
			locationSpace->getObjectsAt(repast::Point<int>(x, y), locationList);
			locationList[0]->setZones(z, mz);
		}
		else {
			goto endloop;
		}
	}
endloop:;
}

void AnasaziModel::readCsvWater()
{
	//read "type","start date","end date","x","y"
	int type, startYear, endYear, x, y;
	string temp;

	std::ifstream file("data/water.csv");//define file object and open water.csv
	file.ignore(500, '\n');//Ignore first line
	while (1)//read until end of file
	{
		getline(file, temp, ',');
		if (!temp.empty())
		{
			getline(file, temp, ',');
			getline(file, temp, ',');
			getline(file, temp, ',');
			type = repast::strToInt(temp); //Read until ',' and convert to int
			getline(file, temp, ',');
			startYear = repast::strToInt(temp); //Read until ',' and convert to int
			getline(file, temp, ',');
			endYear = repast::strToInt(temp); //Read until ',' and convert to int
			getline(file, temp, ',');
			x = repast::strToInt(temp); //Read until ',' and convert to int
			getline(file, temp, '\n');
			y = repast::strToInt(temp); //Read until ',' and convert to int

			std::vector<Location*> locationList;
			locationSpace->getObjectsAt(repast::Point<int>(x, y), locationList);
			locationList[0]->addWaterSource(type, startYear, endYear);
			//locationList[0]->checkWater(existStreams, existAlluvium, x, y, year);
		}
		else
		{
			goto endloop;
		}
	}
endloop:;
}

void AnasaziModel::readCsvPdsi()
{
	//read "year","general","north","mid","natural","upland","kinbiko"
	int i = 0;
	string temp;

	std::ifstream file("data/pdsi.csv");//define file object and open pdsi.csv
	file.ignore(500, '\n');//Ignore first line

	while (1)//read until end of file
	{
		getline(file, temp, ',');
		if (!temp.empty())
		{
			pdsi[i].year = repast::strToInt(temp); //Read until ',' and convert to int
			getline(file, temp, ',');
			pdsi[i].pdsiGeneral = repast::strToDouble(temp); //Read until ',' and convert to double
			getline(file, temp, ',');
			pdsi[i].pdsiNorth = repast::strToDouble(temp); //Read until ',' and convert to double
			getline(file, temp, ',');
			pdsi[i].pdsiMid = repast::strToDouble(temp); //Read until ',' and convert to double
			getline(file, temp, ',');
			pdsi[i].pdsiNatural = repast::strToDouble(temp); //Read until ',' and convert to int
			getline(file, temp, ',');
			pdsi[i].pdsiUpland = repast::strToDouble(temp); //Read until ',' and convert to int
			getline(file, temp, '\n');
			pdsi[i].pdsiKinbiko = repast::strToDouble(temp); //Read until ',' and convert to double
			i++;
		}
		else {
			goto endloop;
		}
	}
endloop:;
}

void AnasaziModel::readCsvHydro()
{
	//read "year","general","north","mid","natural","upland","kinbiko"
	string temp;
	int i = 0;

	std::ifstream file("data/hydro.csv");//define file object and open hydro.csv
	file.ignore(500, '\n');//Ignore first line

	while (1)//read until end of file
	{
		getline(file, temp, ',');
		if (!temp.empty())
		{
			hydro[i].year = repast::strToInt(temp); //Read until ',' and convert to int
			getline(file, temp, ',');
			hydro[i].hydroGeneral = repast::strToDouble(temp); //Read until ',' and convert to double
			getline(file, temp, ',');
			hydro[i].hydroNorth = repast::strToDouble(temp); //Read until ',' and convert to double
			getline(file, temp, ',');
			hydro[i].hydroMid = repast::strToDouble(temp); //Read until ',' and convert to double
			getline(file, temp, ',');
			hydro[i].hydroNatural = repast::strToDouble(temp); //Read until ',' and convert to int
			getline(file, temp, ',');
			hydro[i].hydroUpland = repast::strToDouble(temp); //Read until ',' and convert to int
			getline(file, temp, '\n');
			hydro[i].hydroKinbiko = repast::strToDouble(temp); //Read until ',' and convert to double
			i++;
		}
		else
		{
			goto endloop;
		}
	}
endloop:;
}

int AnasaziModel::yieldFromPdsi(int zone, int maizeZone)
{
	int pdsiValue, row, col;
	switch (zone)
	{
	case 1:
		pdsiValue = pdsi[year - param.startYear].pdsiNatural;
		break;
	case 2:
		pdsiValue = pdsi[year - param.startYear].pdsiKinbiko;
		break;
	case 3:
		pdsiValue = pdsi[year - param.startYear].pdsiUpland;
		break;
	case 4:
	case 6:
		pdsiValue = pdsi[year - param.startYear].pdsiNorth;
		break;
	case 5:
		pdsiValue = pdsi[year - param.startYear].pdsiGeneral;
		break;
	case 7:
	case 8:
		pdsiValue = pdsi[year - param.startYear].pdsiMid;
		break;
	default:
		return 0;
	}

	/* Rows of pdsi table*/
	if (pdsiValue < -3)
	{
		row = 0;
	}
	else if (pdsiValue >= -3 && pdsiValue < -1)
	{
		row = 1;
	}
	else if (pdsiValue >= -1 && pdsiValue < 1)
	{
		row = 2;
	}
	else if (pdsiValue >= 1 && pdsiValue < 3)
	{
		row = 3;
	}
	else if (pdsiValue >= 3)
	{
		row = 4;
	}
	else
	{
		return 0;
	}

	/* Col of pdsi table*/
	if (maizeZone >= 2)
	{
		col = maizeZone - 2;
	}
	else
	{
		return 0;
	}

	return yieldLevels[row][col];
}

double AnasaziModel::hydroLevel(int zone)
{
	switch (zone)
	{
	case 1:
		return hydro[year - param.startYear].hydroNatural;
	case 2:
		return hydro[year - param.startYear].hydroKinbiko;
	case 3:
		return hydro[year - param.startYear].hydroUpland;
	case 4:
	case 6:
		return hydro[year - param.startYear].hydroNorth;
	case 5:
		return hydro[year - param.startYear].hydroGeneral;
	case 7:
	case 8:
		return hydro[year - param.startYear].hydroMid;
	default:
		return 0;
	}
}

void AnasaziModel::checkWaterConditions()
{
	if ((year >= 280 && year < 360) or (year >= 800 && year < 930) or (year >= 1300 && year < 1450))
	{
		existStreams = true;
	}
	else
	{
		existStreams = false;
	}

	if (((year >= 420) && (year < 560)) or ((year >= 630) && (year < 680)) or ((year >= 980) && (year < 1120)) or ((year >= 1180) && (year < 1230)))
	{
		existAlluvium = true;
	}
	else
	{
		existAlluvium = false;
	}
}

void AnasaziModel::writeOutputToFile()
{
	out << year << "," << context.size() << std::endl;
}

void  AnasaziModel::updateLocationProperties()
{
	checkWaterConditions();
	int x = 0;
	for (int i = 0; i < boardSizeX; i++)
	{
		for (int j = 0; j < boardSizeY; j++)
		{
			std::vector<Location*> locationList;
			locationSpace->getObjectsAt(repast::Point<int>(i, j), locationList);
			locationList[0]->checkWater(existStreams, existAlluvium, i, j, year);
			int mz = locationList[0]->getMaizeZone();
			int z = locationList[0]->getZone();
			int y = yieldFromPdsi(z, mz);
			locationList[0]->calculateYield(y, param.harvestAdjustment, yieldGen->next());
		}
	}
}

void AnasaziModel::updateHouseholdProperties()
{
	int noOfAgents = context.size();

	std::vector<repast::AgentId> followers_t;
	std::vector<repast::AgentId> natural_migrants_t;

	//15 std::vector<MigrantRecord>      migrants_records_t;   // 用于 migrantsHistory
	std::vector<MigrantRecord> natural_records_t;   // 只存本年的自然迁出记录

	int highDebtCount_t = 0;  // 本年高债务迁出数（用于打印）

	if (noOfAgents < 1) return;

	std::vector<Household*> agents;
	context.selectAgents(repast::SharedContext<Household>::LOCAL, noOfAgents, agents);
	auto it = agents.begin();

	// ===================== Stage 1：死亡 + 自然迁出 + fission + nextYear =====================
	while (it != agents.end())
	{
		Household* household = (*it);

		if (household->death())
		{
			repast::AgentId hid = household->getId();
			removeHouseholdById(hid);
			++it;
			continue;
		}


		// -------- 自然迁出：checkMaize + fieldSearch --------
		if (!household->checkMaize(param.householdNeed))
		{
			repast::AgentId hid = household->getId();
			std::vector<int> loc;
			householdSpace->getLocation(hid, loc);

			bool fieldFound = fieldSearch(household);

			if (!fieldFound)
			{
				// 记录自然迁出
				natural_migrants_t.push_back(hid);

				MigrantRecord rec;
				rec.id = hid;
				rec.location = loc;
				//15 migrants_records_t.push_back(rec);
				natural_records_t.push_back(rec);   // 只把自然迁出者放进 natural_records_t


				// ★关键：自然迁出必须从 context 移除，否则会继续参与 Stage2/3/4
				removeHouseholdById(hid);

				++it;
				continue;
			}
		}

		// -------- fission --------
		if (household->fission(param.minFissionAge, param.maxFissionAge,
			fissionGen->next(), param.fertilityProbability))
		{
			int rank = repast::RepastProcess::instance()->rank();
			repast::AgentId id(houseID, rank, 2);

			int mStorage = household->splitMaizeStored(param.maizeStorageRatio);
			Household* newAgent = new Household(id, 0, deathAgeGen->next(), mStorage);
			context.addAgent(newAgent);

			// ★亲缘登记：child -> parent（用于 K_ij）
			parentOf[keyOf(newAgent->getId())] = keyOf(household->getId());

			std::vector<int> loc;
			householdSpace->getLocation(household->getId(), loc);
			householdSpace->moveTo(id, repast::Point<int>(loc[0], loc[1]));
			fieldSearch(newAgent);
			houseID++;
		}

		// -------- 原模型：进入下一年（更新 age 和 maizeStorage）--------
		household->nextYear(param.householdNeed);

		++it;
	}
	// ===================== Stage 2：Feature 1 —— 借贷 / 声誉 / Debt_ij / D_i / B_i =====================
	{
		std::vector<Household*> high_debt_households_t;
		processSocialRiskPooling(high_debt_households_t);   // ★里面已算好 D_i/B_i 并更新 debt/rep/maize

		// 先转成 id 列表（不要直接用 Household* 去 remove）
		std::vector<repast::AgentId> highDebtIds;
		highDebtIds.reserve(high_debt_households_t.size());
		for (auto* h : high_debt_households_t) {
			if (h) highDebtIds.push_back(h->getId());
		}

		// 去重：防止同一户重复进入列表 -> 重复删除 -> use-after-free
		std::sort(highDebtIds.begin(), highDebtIds.end(),
			[&](const repast::AgentId& a, const repast::AgentId& b) {
				return keyOf(a) < keyOf(b);
			});

		highDebtIds.erase(std::unique(highDebtIds.begin(), highDebtIds.end(),
			[&](const repast::AgentId& a, const repast::AgentId& b) {
				return keyOf(a) == keyOf(b);
			}), highDebtIds.end());

		highDebtCount_t = (int)highDebtIds.size();

		// 高债务迁出：记录 + 移除（按 id）
		for (const auto& hid : highDebtIds)
		{
			std::vector<int> loc;
			householdSpace->getLocation(hid, loc);   // 移除前取坐标

			/*15 MigrantRecord rec;
			rec.id       = hid;
			rec.location = loc;
			migrants_records_t.push_back(rec);*/

			// ★关键：按 id 删除（内部会 context.getAgent 校验存在性）
			removeHouseholdById(hid);
		}

		// ★重要：不要再做占位版 D/B 循环；processSocialRiskPooling 已经 setDeficitRatio / setBufferRatio
	}


	// ===================== Stage 3：Feature 2 —— A_total,i 和 C_stability,i =====================
	{
		std::vector<Household*> survivors;
		context.selectAgents(repast::SharedContext<Household>::LOCAL, context.size(), survivors);

		for (auto* h : survivors)
		{
			h->incrementYearsInSettlement();

			bool surplus = (h->getMaizeStorage() > 0);
			h->recordYearOutcome(surplus);

			int totalYears = h->getTotalYearsInSettlement();
			int surplusYears = h->getSurplusYears();

			double C = 0.0;
			if (totalYears + settle.a + settle.b > 0.0) {
				C = (static_cast<double>(surplusYears) + settle.a) /
					(static_cast<double>(totalYears) + settle.a + settle.b);
			}
			h->setStability(C);

			int    t = h->getYearsInSettlement();
			double A_time = 1.0 - std::exp(-settle.lambda * static_cast<double>(t));

			// ★现在 tieStrength 已经有了（rep/kin），这里就不要再写 0 了
			double social_sum = 0.0;
			for (auto* other : survivors) {
				if (other == h) continue;
				social_sum += tieStrength(h, other);   // Σ_j S_ij
			}

			double A_total = A_time + settle.alpha * social_sum;
			if (A_total < 0.0) A_total = 0.0;
			if (A_total > 1.0) A_total = 1.0;

			h->setAttachment(A_total);

		}
		logCheckSamples(survivors);
	}



	// ===================== Stage 4：Feature 3 —— 跟随迁移 =====================
	processSocialInfluenceMigration(natural_migrants_t, followers_t);
	{
		std::vector<Household*> survivors;
		context.selectAgents(repast::SharedContext<Household>::LOCAL, context.size(), survivors);

		logCheckSummary(
			(int)natural_migrants_t.size(),
			(int)followers_t.size(),
			highDebtCount_t,                 // 你之前在 Stage2 记录的 highDebtCount_t
			(int)survivors.size()
		);
	}



	// ===================== Stage 5：执行迁移 & 更新迁移记忆 =====================

	// 去重（按 key）
	std::sort(followers_t.begin(), followers_t.end(),
		[&](const repast::AgentId& a, const repast::AgentId& b) {
			return keyOf(a) < keyOf(b);
		});
	followers_t.erase(std::unique(followers_t.begin(), followers_t.end(),
		[&](const repast::AgentId& a, const repast::AgentId& b) {
			return keyOf(a) == keyOf(b);
		}), followers_t.end());

	// 记录 + 按 id 删除
	for (const auto& hid : followers_t)
	{
		std::vector<int> loc;
		householdSpace->getLocation(hid, loc);

		/*15
		MigrantRecord rec;
		rec.id       = hid;
		rec.location = loc;
		migrants_records_t.push_back(rec);*/

		removeHouseholdById(hid);
	}

	//15 migrantsHistory.push_back(migrants_records_t);

	/*15
	if (migrantsHistory.size() > static_cast<size_t>(infl.windowL)) {
		migrantsHistory.erase(migrantsHistory.begin());
	}*/
	//natural_records_t 可能为空（本年没有自然迁出），但也照样 push，这样“过去 L 年”窗口才是连续的。
	//use natural MigrantsHistory of past windowL years)
	naturalMigrantsHistory.push_back(natural_records_t);
	if (naturalMigrantsHistory.size() > static_cast<size_t>(infl.windowL)) {
		naturalMigrantsHistory.erase(naturalMigrantsHistory.begin());
	}


}



bool AnasaziModel::fieldSearch(Household* household)
{
	/******** Choose Field ********/
	std::vector<int> houseIntLocation;
	householdSpace->getLocation(household->getId(), houseIntLocation);
	repast::Point<int> center(houseIntLocation);

	std::vector<Location*> neighbouringLocations;
	std::vector<Location*> checkedLocations;
	repast::Moore2DGridQuery<Location> moore2DQuery(locationSpace);

	int maxRangeX = std::max(std::abs(houseIntLocation[0] - boardSizeX), houseIntLocation[0]);
	int maxRangeY = std::max(std::abs(houseIntLocation[1] - boardSizeY), houseIntLocation[1]);
	int maxRange = std::max(maxRangeX, maxRangeY);

	int range = 1;
	while (1)
	{
		moore2DQuery.query(houseIntLocation, range, false, neighbouringLocations);

		for (std::vector<Location*>::iterator it = neighbouringLocations.begin(); it != neighbouringLocations.end(); ++it)
		{
			Location* tempLoc = (&**it);
			if (std::find(checkedLocations.begin(), checkedLocations.end(), tempLoc) != checkedLocations.end())
			{
				continue;
			}
			checkedLocations.push_back(tempLoc);

			if (tempLoc->getState() == 0)
			{
				if (tempLoc->getExpectedYield() >= param.householdNeed)
				{
					std::vector<int> loc;
					locationSpace->getLocation(tempLoc->getId(), loc);
					household->chooseField(tempLoc);
					goto EndOfLoop;
				}
			}
		}
		neighbouringLocations.clear();
		range++;
		if (range > maxRange)
		{
			return false;
		}
	}
EndOfLoop:
	if (range >= 10)
	{
		return relocateHousehold(household);
	}
	else
	{
		return true;
	}
}

void AnasaziModel::removeHousehold(Household* household)
{
	static int dbgPtr = 0;
	if (dbgPtr < 50) {
		std::cerr << "[removeHousehold] ptr=" << household << std::endl;
		dbgPtr++;
	}
	repast::AgentId id = household->getId();

	// --- 防重复删除：如果 agent 已经不在 context，直接返回 ---
	// Repast HPC context 有 getAgent，若你没有这个接口，就先跳过这段
	Household* chk = context.getAgent(id);
	if (chk == NULL) return;

	// --- 边界检查函数（确保不会用越界坐标访问 grid）---
	auto inBounds = [&](int x, int y) {
		return (x >= 0 && y >= 0 && x < boardSizeX && y < boardSizeY);
		};

	// --- 取房屋坐标 ---
	std::vector<int> houseLoc;
	householdSpace->getLocation(id, houseLoc);

	if (houseLoc.size() >= 2 && inBounds(houseLoc[0], houseLoc[1]))
	{
		repast::Point<int> hPt(houseLoc[0], houseLoc[1]);

		Location* householdLocation = locationSpace->getObjectAt(hPt);

		std::vector<Household*> householdList;
		householdSpace->getObjectsAt(hPt, householdList);

		if (householdLocation != NULL && householdList.size() == 1) {
			householdLocation->setState(0);
		}
	}

	// --- assignedField: 这里最容易是悬空指针，所以要“先拿 id 再用” + 也要边界检查 ---
	Location* af = household->getAssignedField();
	if (af != NULL)
	{
		// ⚠️ 如果 af 是悬空指针，下面这句也可能崩。
		// 所以建议：把 assignedField 改成保存 AgentId 或 key，而不是裸指针。
		// 但短期先加 try/catch 不行（C++ 段错误抓不到）。
		// 我们用更保守做法：先从 locationSpace 查询它的位置（如果查不到就跳过）

		std::vector<int> fieldLoc;
		locationSpace->getLocation(af->getId(), fieldLoc);

		if (fieldLoc.size() >= 2 && inBounds(fieldLoc[0], fieldLoc[1]))
		{
			repast::Point<int> fPt(fieldLoc[0], fieldLoc[1]);
			Location* fieldLocation = locationSpace->getObjectAt(fPt);
			if (fieldLocation != NULL) fieldLocation->setState(0);
		}
	}

	// ===== debt / kin 清理 =====
	std::uint64_t k = keyOf(id);

	debt_ij.erase(k);

	for (auto it = debt_ij.begin(); it != debt_ij.end(); )
	{
		it->second.erase(k);
		if (it->second.empty()) it = debt_ij.erase(it);
		else ++it;
	}

	parentOf.erase(k);
	for (auto it = parentOf.begin(); it != parentOf.end(); )
	{
		if (it->second == k) it = parentOf.erase(it);
		else ++it;
	}

	// --- 最后再 remove ---
	context.removeAgent(id);
}



bool AnasaziModel::relocateHousehold(Household* household)
{
	std::vector<Location*> fieldNeighbouringLocations;
	std::vector<Location*> suitableHouseholdLocations;
	std::vector<Location*> waterSources;

	std::vector<int> fieldIntLocation, houseIntLocation;
	// get the location of the field and household
	locationSpace->getLocation(household->getAssignedField()->getId(), fieldIntLocation);
	householdSpace->getLocation(household->getId(), houseIntLocation);

	Location* householdLocation = locationSpace->getObjectAt(repast::Point<int>(houseIntLocation));

	int maxRangeX = std::max(std::abs(fieldIntLocation[0] - boardSizeX), fieldIntLocation[0]);
	int maxRangeY = std::max(std::abs(fieldIntLocation[1] - boardSizeY), fieldIntLocation[1]);
	int maxRange = std::max(maxRangeX, maxRangeY);

	repast::Moore2DGridQuery<Location> moore2DQuery(locationSpace);
	int range = floor(param.maxDistance / 100);
	int i = 1;
	bool conditionC = true;

	std::vector<Location*> checkedHouseLocations;

	//get all !Field with 1km
LocationSearch:
	moore2DQuery.query(fieldIntLocation, range * i, false, fieldNeighbouringLocations);
	for (std::vector<Location*>::iterator it = fieldNeighbouringLocations.begin(); it != fieldNeighbouringLocations.end(); ++it)
	{
		Location* tempHouseLoc = (&**it);
		if (std::find(checkedHouseLocations.begin(), checkedHouseLocations.end(), tempHouseLoc) != checkedHouseLocations.end())
		{
			continue;
		}
		checkedHouseLocations.push_back(tempHouseLoc);

		if (tempHouseLoc->getState() != 2)
		{
			if (householdLocation->getExpectedYield() < tempHouseLoc->getExpectedYield() && conditionC == true)
			{
				suitableHouseholdLocations.push_back(tempHouseLoc);
			}
			if (tempHouseLoc->getWater())
			{
				waterSources.push_back(tempHouseLoc);
			}
		}
	}
	fieldNeighbouringLocations.clear();
	if (suitableHouseholdLocations.size() == 0 || waterSources.size() == 0)
	{
		if (conditionC == true)
		{
			conditionC = false;
		}
		else
		{
			conditionC = true;
			i++;
			if (range * i > maxRange)
			{
				removeHousehold(household);
				return false;
			}
		}
		goto LocationSearch;
	}
	else if (suitableHouseholdLocations.size() == 1)
	{
		std::vector<int> futureHouseIntLocation;
		locationSpace->getLocation(suitableHouseholdLocations[0]->getId(), futureHouseIntLocation);
		householdSpace->moveTo(household->getId(), repast::Point<int>(futureHouseIntLocation));
		household->resetSettlementClock();   // 换聚落后重新计数
		return true;
	}
	else
	{
		std::vector<int> point1, point2;
		std::vector<double> minDistances;
		for (std::vector<Location*>::iterator it1 = suitableHouseholdLocations.begin(); it1 != suitableHouseholdLocations.end(); ++it1)
		{
			locationSpace->getLocation((&**it1)->getId(), point1);
			std::vector<double> distances;
			for (std::vector<Location*>::iterator it2 = waterSources.begin(); it2 != waterSources.end(); ++it2)
			{
				locationSpace->getLocation((&**it2)->getId(), point2);
				double distance = sqrt(pow((point1[0] - point2[0]), 2) + pow((point1[1] - point2[1]), 2));
				distances.push_back(distance);
			}
			minDistances.push_back(*std::min_element(distances.begin(), distances.end()));
		}

		// Select the household location with the closest water source
		int minElementIndex = std::min_element(minDistances.begin(), minDistances.end()) - minDistances.begin();
		std::vector<int> futureHouseIntLocation;
		locationSpace->getLocation(suitableHouseholdLocations[minElementIndex]->getId(), futureHouseIntLocation);
		householdSpace->moveTo(household->getId(), repast::Point<int>(futureHouseIntLocation));
		household->resetSettlementClock();   // 换聚落后重新计数
		return true;
	}


}

//component4 added
double AnasaziModel::distanceBetween(const Household* a, const Household* b) const {
	std::vector<int> locA, locB;
	householdSpace->getLocation(a->getId(), locA);
	householdSpace->getLocation(b->getId(), locB);
	if (locA.size() < 2 || locB.size() < 2) return 0.0;

	double dx = static_cast<double>(locA[0] - locB[0]);
	double dy = static_cast<double>(locA[1] - locB[1]);
	return std::sqrt(dx * dx + dy * dy);
}

Household* AnasaziModel::getHouseholdById(const repast::AgentId& id) {
	return context.getAgent(id);
}

// ===== 下面四个暂时用占位实现 =====
// 你实现 Feature 1 & 2 后，把这些改成真正的字段 / 公式

double AnasaziModel::getDeficitRatio(const Household* h) const {
	// TODO: 替换为 D_i（例如 h->getDeficitRatio()）
	return 0.0;
}

double AnasaziModel::getBufferingRatio(const Household* h) const {
	// TODO: 替换为 B_i（例如 h->getBufferingRatio()）
	return 0.0;
}

double AnasaziModel::getAttachment(const Household* h) const {
	// TODO: 替换为 A_total_i（例如 h->getAttachment()）
	return 0.0;
}

double AnasaziModel::getStability(const Household* h) const {
	// TODO: 替换为 Cstability_i（例如 h->getStability()）
	return 0.0;
}

//15只用自然迁出者位置来累计距离衰减
double AnasaziModel::computeExposure(Household* i)
{
	if (infl.windowL <= 0) return 0.0;

	// 过去 windowL 年自然迁出总人数为 0 => exposure=0
	int nat_sum = 0;
	for (const auto& yr : naturalMigrantsHistory) nat_sum += (int)yr.size();
	if (nat_sum == 0) return 0.0;

	std::vector<int> loc_i;
	householdSpace->getLocation(i->getId(), loc_i);
	if (loc_i.size() < 2) return 0.0;

	double E = 0.0;

	// 只用自然迁出者位置来累计距离衰减
	for (const auto& yearRecords : naturalMigrantsHistory) {
		for (const auto& rec : yearRecords) {
			if (rec.location.size() < 2) continue;
			if (rec.id == i->getId())    continue;

			double dx = static_cast<double>(loc_i[0] - rec.location[0]);
			double dy = static_cast<double>(loc_i[1] - rec.location[1]);
			double d = std::sqrt(dx * dx + dy * dy);

			E += std::exp(-infl.rho * d);
		}
	}
	return E;
}






void AnasaziModel::processSocialInfluenceMigration(
	const std::vector<repast::AgentId>& natural_migrants_t,
	std::vector<repast::AgentId>& followers_t)
{
	int nat_sum = 0;
	for (const auto& yr : naturalMigrantsHistory) nat_sum += (int)yr.size();
	if (nat_sum == 0) return;


	// 当前仍在情境中的全部家庭
	std::vector<Household*> all;
	context.selectAgents(repast::SharedContext<Household>::LOCAL, context.size(), all);

	// 把自然迁出者 ID 做成一个集合，避免他们再被当成跟随候选
	std::vector<repast::AgentId> nat_ids = natural_migrants_t;

	for (auto h : all) {

		// 若是本年自然迁出者，则不参与跟随决策
		if (std::find(nat_ids.begin(), nat_ids.end(), h->getId()) != nat_ids.end())
			continue;

		double E = computeExposure(h);

		if (followMigrationDecision(h, E)) {
			followers_t.push_back(h->getId());
		}
	}

	// 年度 cap：|followers_t| ≤ cap * N_t
	int N_t = static_cast<int>(all.size());  // 这里近似用当前人口
	int capFollow = static_cast<int>(infl.capPerYear * N_t);

	if (capFollow < 0) capFollow = 0;
	if (capFollow < static_cast<int>(followers_t.size())) {
		followers_t.resize(capFollow);
	}
}





//new add:followMigrationDecision → 使用 z_i = k_d D − k_b B − k_a A − k_c C + k_e E
bool AnasaziModel::followMigrationDecision(const Household* h, double E)
{
	double D = h->getDeficitRatio();   // D_i
	double B = h->getBufferRatio();    // B_i
	double A = h->getAttachment();     // A_total,i
	double C = h->getStability();      // C_stability,i

	double z = infl.k_d * D
		- infl.k_b * B
		- infl.k_a * A
		- infl.k_c * C
		+ infl.k_e * E;

	double p = 1.0 / (1.0 + std::exp(-z));

	// 按文档：p_i > probability_base 即跟随迁出（不再额外抽随机数）
	return (p > infl.probBase);
}
static void ensureOutputsDirExists() {
#ifdef _WIN32
	_mkdir("outputs");                 // 已存在会失败，但没关系
#else
	mkdir("outputs", 0755);            // 已存在会失败，但没关系
#endif
}

void AnasaziModel::logCheckSummary(int naturalCount, int followerCount, int highDebtCount, int survivorsCount)
{
	ensureOutputsDirExists();

	int rank = repast::RepastProcess::instance()->rank();
	if (repast::RepastProcess::instance()->rank() != 0) return;
	std::string path = "outputs/check_summary_rank" + std::to_string(rank) + ".csv";
	static std::ofstream out;
	static bool opened = false;

	if (!opened) {
		out.open(path, std::ios::app);
		// 如果文件是新建的，写表头（简单做法：当文件刚打开且位置在 0）
		if (out.tellp() == 0) {
			out << "year,natural,followers,highDebt,survivors\n";
		}
		opened = true;
	}

	out << year << ","
		<< naturalCount << ","
		<< followerCount << ","
		<< highDebtCount << ","
		<< survivorsCount
		<< "\n";
	out.flush();
}

void AnasaziModel::logCheckSamples(const std::vector<Household*>& survivors)
{
	if (survivors.empty()) return;
	ensureOutputsDirExists();

	int rank = repast::RepastProcess::instance()->rank();
	if (repast::RepastProcess::instance()->rank() != 0) return;
	std::string path = "outputs/check_samples_rank" + std::to_string(rank) + ".csv";
	static std::ofstream out;
	static bool opened = false;

	if (!opened) {
		out.open(path, std::ios::app);
		if (out.tellp() == 0) {
			out << "year,hh_id,x,y,storage,t,D,B,A,C,E,z,p\n";
		}
		opened = true;
	}

	int n = (int)survivors.size();
	int k = 1;// 15 只采 1 户

	repast::IntUniformGenerator pick(repast::Random::instance()->createUniIntGenerator(0, n - 1));

	int idx1 = pick.next();
	int idx2 = -1;

	if (n >= 2) {
		idx2 = pick.next();
		if (idx2 == idx1) idx2 = (idx1 + 1) % n;
	}


	auto writeOne = [&](Household* h) {
		double D = h->getDeficitRatio();
		double B = h->getBufferRatio();
		double A = h->getAttachment();
		double C = h->getStability();
		double E = computeExposure(h);

		double z = infl.k_d * D
			- infl.k_b * B
			- infl.k_a * A
			- infl.k_c * C
			+ infl.k_e * E;

		double p = 1.0 / (1.0 + std::exp(-z));

		std::vector<int> loc;
		householdSpace->getLocation(h->getId(), loc);
		int x = (loc.size() >= 2) ? loc[0] : -1;
		int y = (loc.size() >= 2) ? loc[1] : -1;

		out << year << ","
			<< h->getId().id() << ","
			<< x << ","
			<< y << ","
			<< h->getMaizeStorage() << ","
			<< h->getYearsInSettlement() << ","
			<< std::setprecision(10)
			<< D << ","
			<< B << ","
			<< A << ","
			<< C << ","
			<< E << ","
			<< z << ","
			<< p
			<< "\n";
		};

	writeOne(survivors[idx1]);
	if (idx2 >= 0) writeOne(survivors[idx2]);

	out.flush();
}

void AnasaziModel::removeHouseholdById(const repast::AgentId& id)
{
	Household* h = context.getAgent(id);
	if (h == nullptr) return;      // 已经删过 / 不存在：直接跳过
	removeHousehold(h);            // 只在这里用指针，且一定是 context 里现取的“活指针”
}



