#ifndef NEXUSMINER_MINING_CLIENT_CHANNEL_MANAGER_H
#define NEXUSMINER_MINING_CLIENT_CHANNEL_MANAGER_H

#include "client_block_state.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace nexusminer {
namespace mining {

/**
 * @brief ClientChannelManager - CLIENT-SIDE equivalent of NODE's ChannelStateManager
 * 
 * Manages channel-specific state for mining, tracking heights and detecting forks.
 * Validates templates using same logic as NODE's Block::Accept().
 * 
 * Key Features:
 * - Tracks unified + channel heights (from GET_ROUND)
 * - Detects forks (height regression)
 * - Validates templates (dual height check)
 * - Handles rollbacks (clears invalid templates)
 * 
 * Architecture Alignment:
 *   NODE: ChannelStateManager
 *   CLIENT: ClientChannelManager (this class)
 * 
 * INTEGRATION NOTE:
 * This class provides height tracking and fork detection capabilities. It CAN store
 * templates (m_pCurrentTemplate) for standalone use or testing, but in the current
 * Solo protocol integration, template management is delegated to MiningTemplateInterface.
 * 
 * Current Integration Pattern (in Solo protocol):
 * - MiningTemplateInterface: Handles actual template storage and feeding to workers
 * - ClientChannelManager: Provides height tracking, fork detection, and validation
 * - Managers update heights from GET_ROUND responses
 * - Managers detect forks and trigger template invalidation in MiningTemplateInterface
 * 
 * The template storage in ClientChannelManager exists to:
 * 1. Mirror NODE's ChannelStateManager architecture completely
 * 2. Support standalone testing (see client_channel_manager_test.cpp)
 * 3. Allow future alternative integrations if needed
 * 
 * For production Solo mining, use MiningTemplateInterface for template management
 * and ClientChannelManager for height/fork tracking.
 */
class ClientChannelManager
{
protected:
    const uint32_t m_nChannel;      // Channel number (1=Prime, 2=Hash)
    
    // Current node state (from GET_ROUND)
    std::atomic<uint32_t> m_nNodeUnifiedHeight;
    std::atomic<uint32_t> m_nNodeChannelHeight;
    
    // Previous state (for fork detection)
    std::atomic<uint32_t> m_nPrevUnifiedHeight;
    std::atomic<uint32_t> m_nPrevChannelHeight;
    
    // Fork detection
    std::atomic<bool> m_fForkDetected;
    // Phantom Stake regression: 1-block unified height decrease caused by
    // cross-channel tip normalisation (NOT a real fork — no template discard).
    std::atomic<bool> m_fPhantomStakeRegression;
    
    // Current template (protected by mutex)
    std::unique_ptr<ClientBlockState> m_pCurrentTemplate;
    mutable std::mutex m_templateMutex;
    
public:
    /**
     * @brief Constructor
     * @param nChannel Channel number (1=Prime, 2=Hash)
     */
    explicit ClientChannelManager(uint32_t nChannel)
        : m_nChannel(nChannel)
        , m_nNodeUnifiedHeight(0)
        , m_nNodeChannelHeight(0)
        , m_nPrevUnifiedHeight(0)
        , m_nPrevChannelHeight(0)
        , m_fForkDetected(false)
        , m_fPhantomStakeRegression(false)
        , m_pCurrentTemplate(nullptr)
    {
    }
    
    /**
     * @brief Virtual destructor
     */
    virtual ~ClientChannelManager() = default;
    
    // Disable copy/move
    ClientChannelManager(const ClientChannelManager&) = delete;
    ClientChannelManager& operator=(const ClientChannelManager&) = delete;
    ClientChannelManager(ClientChannelManager&&) = delete;
    ClientChannelManager& operator=(ClientChannelManager&&) = delete;
    
    /**
     * @brief Update from GET_ROUND response (CLIENT-SIDE equivalent of SyncWithBlockchain)
     * 
     * Called when GET_ROUND response is received from node.
     * Updates height tracking and detects forks via height regression.
     * 
     * @param nUnified Current unified blockchain height
     * @param nChannel Current channel-specific height
     */
    void UpdateFromGetRound(uint32_t nUnified, uint32_t nChannel)
    {
        uint32_t nPrevUnified = m_nNodeUnifiedHeight.load();
        uint32_t nPrevChannel = m_nNodeChannelHeight.load();
        
        // FORK DETECTION: Only a 2+ block rollback is a genuine fork.
        // A 1-block regression is a "Phantom Stake regression" caused by
        // cross-channel tip normalisation (Stake block advances unified by 1,
        // then GET_ROUND returns the same normalised tip — not a real fork).
        if (nPrevUnified > 0 && nUnified < nPrevUnified)
        {
            uint32_t nRollback = nPrevUnified - nUnified;
            if (nRollback > 1)
            {
                // TRUE FORK: 2+ block rollback
                m_fForkDetected.store(true);
                OnForkDetected(nPrevUnified, nUnified, nRollback);
            }
            else
            {
                // PHANTOM STAKE: 1-block cross-channel tip oscillation — not a fork
                m_fPhantomStakeRegression.store(true);
                OnPhantomStakeRegression(nPrevUnified, nUnified);
            }
        }
        
        // Update previous heights (for next comparison)
        m_nPrevUnifiedHeight.store(nPrevUnified);
        m_nPrevChannelHeight.store(nPrevChannel);
        
        // Update current heights
        m_nNodeUnifiedHeight.store(nUnified);
        m_nNodeChannelHeight.store(nChannel);
    }
    
    /**
     * @brief Validate template using dual height check (mirrors NODE's Block::Accept logic)
     * 
     * Checks:
     * 1. Unified height match (template.nHeight == nodeUnifiedHeight + 1)
     * 2. Channel height directional guard (template.nChannelHeight > nodeChannelHeight)
     * 3. Age timeout (< MAX_TEMPLATE_AGE_SECONDS)
     * 
     * Note: template.nHeight is the UNIFIED blockchain height (tStateBest.nHeight + 1).
     * Channel-specific height is tracked in template.nChannelHeight (metadata only, defensive guard).
     * The channel height check uses a directional guard (not exact equality) so that templates
     * targeting a block more than 1 ahead of the local tracker (normal during burst recovery) are
     * not incorrectly rejected.
     * 
     * @param pTemplate Template to validate
     * @return true if template is valid for mining
     */
    bool ValidateTemplate(const ClientBlockState* pTemplate) const
    {
        if (!pTemplate)
            return false;
        
        uint32_t nNodeUnified = m_nNodeUnifiedHeight.load();
        uint32_t nNodeChannel = m_nNodeChannelHeight.load();
        
        // Validate unified height (Block::Accept logic)
        // Template builds NEXT block; nHeight = unified tip + 1 (nHeight is unified height)
        if (pTemplate->nHeight != nNodeUnified + 1)
            return false;  // Stale template or fork
        
        // Validate channel height (secondary staleness guard — directional check).
        // Stale only when the chain tip has REACHED or PASSED the template's target height.
        // Exact equality (nChannelHeight == nNodeChannel + 1) is not required: during burst
        // recovery the node may hand us a template targeting nNodeChannel + 3 and that is valid.
        if (pTemplate->nChannelHeight <= nNodeChannel)
            return false;  // Chain has already reached or passed our target — stale
        
        // Age timeout (MAX_TEMPLATE_AGE_SECONDS safety net)
        if (pTemplate->GetAge() > MAX_TEMPLATE_AGE_SECONDS)
            return false;  // Too old
        
        return true;  // FRESH
    }
    
    /**
     * @brief Set current template
     * @param pTemplate Template to set (ownership transferred)
     */
    void SetCurrentTemplate(std::unique_ptr<ClientBlockState> pTemplate)
    {
        std::lock_guard<std::mutex> lock(m_templateMutex);
        m_pCurrentTemplate = std::move(pTemplate);
    }
    
    /**
     * @brief Get current template (read-only)
     * @return Pointer to current template, nullptr if none
     */
    const ClientBlockState* GetCurrentTemplate() const
    {
        std::lock_guard<std::mutex> lock(m_templateMutex);
        return m_pCurrentTemplate.get();
    }
    
    /**
     * @brief Get mutable current template (for modifications)
     * @return Pointer to current template, nullptr if none
     */
    ClientBlockState* GetCurrentTemplate()
    {
        std::lock_guard<std::mutex> lock(m_templateMutex);
        return m_pCurrentTemplate.get();
    }
    
    /**
     * @brief Check if valid template exists
     * @return true if template exists
     */
    bool HasValidTemplate() const
    {
        std::lock_guard<std::mutex> lock(m_templateMutex);
        return m_pCurrentTemplate != nullptr;
    }
    
    /**
     * @brief Clear current template
     */
    void ClearTemplate()
    {
        std::lock_guard<std::mutex> lock(m_templateMutex);
        m_pCurrentTemplate.reset();
    }
    
    /**
     * @brief Get current node heights
     * @return Pair of (unified height, channel height)
     */
    std::pair<uint32_t, uint32_t> GetNodeHeights() const
    {
        return std::make_pair(
            m_nNodeUnifiedHeight.load(),
            m_nNodeChannelHeight.load()
        );
    }
    
    /**
     * @brief Get previous node heights (before last update)
     * @return Pair of (previous unified height, previous channel height)
     */
    std::pair<uint32_t, uint32_t> GetPreviousHeights() const
    {
        return std::make_pair(
            m_nPrevUnifiedHeight.load(),
            m_nPrevChannelHeight.load()
        );
    }
    
    /**
     * @brief Get expected heights for next block
     * @return Pair of (expected unified, expected channel)
     */
    std::pair<uint32_t, uint32_t> GetExpectedHeights() const
    {
        return std::make_pair(
            m_nNodeUnifiedHeight.load() + 1,
            m_nNodeChannelHeight.load() + 1
        );
    }
    
    /**
     * @brief Get channel number
     * @return Channel number
     */
    uint32_t GetChannel() const
    {
        return m_nChannel;
    }
    
    /**
     * @brief Get channel name
     * @return Human-readable channel name
     */
    virtual const char* GetChannelName() const
    {
        switch (m_nChannel) {
            case CHANNEL_PRIME: return "PRIME";
            case CHANNEL_HASH: return "HASH";
            case CHANNEL_STAKE: return "STAKE";
            default: return "UNKNOWN";
        }
    }
    
    /**
     * @brief Check if fork was detected
     * @return true if fork detected
     */
    bool IsForkDetected() const
    {
        return m_fForkDetected.load();
    }

    /**
     * @brief Check if a Phantom Stake regression was detected (1-block unified
     *        height decrease due to cross-channel tip normalisation — NOT a fork)
     * @return true if phantom stake regression detected
     */
    bool IsPhantomStakeRegression() const
    {
        return m_fPhantomStakeRegression.load();
    }
    
    /**
     * @brief Clear fork flag and phantom stake regression flag
     *
     * Both detection flags are cleared together since they are mutually
     * exclusive states that represent distinct outcomes of UpdateFromGetRound.
     */
    void ClearForkFlag()
    {
        m_fForkDetected.store(false);
        m_fPhantomStakeRegression.store(false);
    }
    
protected:
    /**
     * @brief Fork detection callback (virtual for subclass customization)
     * 
     * Called when height regression is detected. Default behavior clears template.
     * Subclasses can override for custom fork handling.
     * 
     * @param nPrevHeight Previous unified height
     * @param nNewHeight New (regressed) unified height
     * @param nRollback Number of blocks rolled back
     */
    virtual void OnForkDetected(uint32_t nPrevHeight, uint32_t nNewHeight, uint32_t nRollback)
    {
        // Log fork detection
        // Note: Can't log here directly as we don't have logger access
        // Logging should be done by the calling code
        
        // Clear invalid template
        ClearTemplate();
    }

    /**
     * @brief Phantom Stake regression callback (virtual for subclass customization)
     *
     * Called when a 1-block unified height decrease is detected due to
     * cross-channel tip normalisation timing (Phantom Stake).  This is NOT a
     * real fork — the template should be preserved.  Subclasses (e.g. Solo) can
     * override to add debug/info logging.
     *
     * @param nPrevHeight Previous unified height
     * @param nNewHeight  New (slightly lower) unified height
     */
    virtual void OnPhantomStakeRegression(uint32_t /*nPrevHeight*/, uint32_t /*nNewHeight*/)
    {
        // Default: no action — template is preserved.
    }
};

/**
 * @brief PrimeClientManager - Prime channel (CHANNEL_PRIME) manager
 * 
 * Architecture Alignment:
 *   NODE: PrimeStateManager
 *   CLIENT: PrimeClientManager (this class)
 */
class PrimeClientManager : public ClientChannelManager
{
public:
    PrimeClientManager() 
        : ClientChannelManager(CHANNEL_PRIME) 
    {
    }
    
    const char* GetChannelName() const override 
    { 
        return "PRIME"; 
    }
};

/**
 * @brief HashClientManager - Hash channel (CHANNEL_HASH) manager
 * 
 * Architecture Alignment:
 *   NODE: HashStateManager
 *   CLIENT: HashClientManager (this class)
 */
class HashClientManager : public ClientChannelManager
{
public:
    HashClientManager() 
        : ClientChannelManager(CHANNEL_HASH) 
    {
    }
    
    const char* GetChannelName() const override 
    { 
        return "HASH"; 
    }
};

} // namespace mining
} // namespace nexusminer

#endif // NEXUSMINER_MINING_CLIENT_CHANNEL_MANAGER_H
