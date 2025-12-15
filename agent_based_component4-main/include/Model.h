#ifndef MODEL
#define MODEL

#include <boost/mpi.hpp>
#include "repast_hpc/Schedule.h"
#include "repast_hpc/Properties.h"
#include "repast_hpc/SharedContext.h"
#include "repast_hpc/SharedDiscreteSpace.h"
#include "repast_hpc/GridComponents.h"
#include "repast_hpc/Random.h"
#include <math.h>
#include <vector>
#include <fstream>
#include <unordered_map>
#include "Household.h"

#define NUMBER_OF_YEARS 551

class AnasaziModel{
	
	// ================= PUBLIC STRUCTS =================
	public: 
		// 将 MigrantRecord 移到 public，以便作为函数参数类型
		struct MigrantRecord {
			repast::AgentId id;
			std::vector<int> location;   // {x, y}
		};

	private:
		int year;
		int stopAt;
		int boardSizeX, boardSizeY, procX, procY, bufferSize;
		int randomSeed;
		int houseID = 0;

		std::ofstream out;
		
		struct Parameters
		{
			int startYear;
			int endYear;
			int maxStorageYear;
			int maxStorage;
			int householdNeed;
			int minFissionAge;
			int maxFissionAge;
			int minDeathAge;
			int maxDeathAge;
			int maxDistance;
			int initMinCorn;
			int initMaxCorn;
			double annualVariance;
			double spatialVariance;
			double fertilityProbability;
			double harvestAdjustment;
			double maizeStorageRatio;
		} param;

		struct PDSI
		{
			int year;
			double pdsiGeneral;
			double pdsiNorth;
			double pdsiMid;
			double pdsiNatural;
			double pdsiUpland;
			double pdsiKinbiko;
		} pdsi[NUMBER_OF_YEARS];

		struct Hydro
		{
			int year;
			double hydroGeneral;
			double hydroNorth;
			double hydroMid;
			double hydroNatural;
			double hydroUpland;
			double hydroKinbiko;
		} hydro[NUMBER_OF_YEARS];

		// Feature 3: Migration Influence Params
		struct InfluenceParams {
			int    windowL;       // L：记忆年数
			double rho;           // 影响衰减
			double capPerYear;    // 每年最多跟随比例（cap）
			double k_d;
			double k_b;
			double k_a;
			double k_c;
			double k_e;
			double probBase;      // probability_base
		};
		InfluenceParams infl;

		// Feature 2: Settlement attachment params
		struct SettlementParams {
			double lambda;   // settlement.lambda
			double alpha;    // A_total_i 中 α
			double a;        // Laplace 平滑 a
			double b;        // Laplace 平滑 b
		};
		SettlementParams settle;

		// Feature 1: Social risk-pooling params
		struct SocialParams {
			double ks;       // kinship 权重 k_s
			double k_up;     // 声誉增加
			double k_down;   // 声誉惩罚
		};
		SocialParams social;

		// 历史记录（用于计算暴露度 E_i，虽然最新逻辑只用当年的，但这里保留结构以防扩展）
		std::vector<std::vector<MigrantRecord>> migrantsHistory;

		const int yieldLevels[5][4] = { {617, 514, 411, 642},
			{719, 599, 479, 749},
			{821, 684, 547, 855},
			{988, 824, 659, 1030},
			{1153, 961, 769, 1201}};

		bool existStreams;
		bool existAlluvium;
		
		repast::Properties* props;
		repast::SharedContext<Household> context;
		repast::SharedContext<Location> locationContext;
		repast::SharedDiscreteSpace<Household, repast::StrictBorders, repast::SimpleAdder<Household> >* householdSpace;
		repast::SharedDiscreteSpace<Location, repast::StrictBorders, repast::SimpleAdder<Location> >* locationSpace;
		
		repast::DoubleUniformGenerator* fissionGen;
		repast::IntUniformGenerator* deathAgeGen;
		repast::NormalGenerator* yieldGen;
		repast::NormalGenerator* soilGen;
		repast::IntUniformGenerator* initAgeGen;
		repast::IntUniformGenerator* initMaizeGen;

		// ---------- Stage 2 storage & helpers ----------
		std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, double>> debt_ij; // borrower -> (lender -> amount)
		std::unordered_map<std::uint64_t, std::uint64_t> parentOf; // child -> parent

		// Helpers
		double distanceBetween(const Household* a, const Household* b) const;
		Household* getHouseholdById(const repast::AgentId& id);
		std::uint64_t keyOf(const repast::AgentId& id) const;
		bool isKin(std::uint64_t a, std::uint64_t b) const;
		double clamp01(double x) const;
		double tieStrength(Household* borrower, Household* lender) const;   // S_ij
		double randn01() const;
		double truncNormal(double mu, double sigma, double lo, double hi, int maxTry=20) const;

		// Stage 2 main process
		void processSocialRiskPooling(std::vector<Household*>& highDebtHouseholds);

		// CSV Logging
		void logCheckSummary(int naturalCount, int followerCount, int highDebtCount, int survivorsCount);
		void logCheckSamples(const std::vector<Household*>& survivors);

		// Feature 3: Migration calculation
		// 修改点：只接收当年的自然迁出列表
		double computeExposure(Household* i, const std::vector<MigrantRecord>& current_natural_migrants);

	public:
		AnasaziModel(std::string propsFile, int argc, char** argv, boost::mpi::communicator* comm);
		~AnasaziModel();
		
		void initAgents();
		void initSchedule(repast::ScheduleRunner& runner);
		void doPerTick();
		
		void readCsvMap();
		void readCsvWater();
		void readCsvPdsi();
		void readCsvHydro();
		
		int yieldFromPdsi(int zone, int maizeZone);
		double hydroLevel(int zone);
		void checkWaterConditions();
		void writeOutputToFile();
		
		void updateLocationProperties();
		void updateHouseholdProperties();
		
		bool fieldSearch(Household* household);
		void removeHousehold(Household* household);
		void removeHouseholdById(const repast::AgentId& id);
		bool relocateHousehold(Household* household);

		// Feature 3: Follow-migration decision
		bool followMigrationDecision(const Household* h, double E);

		// 修改点：接收当年的记录列表进行处理
		void processSocialInfluenceMigration(
				const std::vector<MigrantRecord>& natural_migrants_records,
				std::vector<repast::AgentId>& followers_t);
};

#endif