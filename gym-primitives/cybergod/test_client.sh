ADMIN_PUBKEY=...
ADMIN_PRIVKEY=...

USER1_PUBKEY=...
USER1_PRIVKEY=...

USER2_PUBKEY=...
USER2_PRIVKEY=...

echo "Create an account (user1)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" create_account --new-public-key "$USER1_PUBKEY"

echo "Create an account (user2)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" create_account --new-public-key "$USER2_PUBKEY" --tax-free

echo "Mint credits (user1)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" mint --account "$USER1_PUBKEY" --amount 100

echo "Mint credits (user2)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" mint --account "$USER2_PUBKEY" --amount 100

echo "Burn credits (user1)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" burn --account "$USER1_PUBKEY" --amount 10

echo "Check balance (user1)"
python bank-client.py --private-key "$USER1_PRIVKEY" balance --account "$USER1_PUBKEY"

echo "Check balance (user2)"
python bank-client.py --private-key "$USER2_PRIVKEY" balance --account "$USER2_PUBKEY"

echo "Transfer credits (user1 -> user2)"
python bank-client.py --private-key "$USER1_PRIVKEY" transfer --to-account "$USER2_PUBKEY" --amount 50

echo "Check tax (user1)"
python bank-client.py --private-key "$USER1_PRIVKEY" check_tax --account "$USER1_PUBKEY"

echo "Check tax (user2)"
python bank-client.py --private-key "$USER2_PRIVKEY" check_tax --account "$USER2_PUBKEY"

echo "Check account info (user2)"
python bank-client.py --private-key "$USER2_PRIVKEY" info --account "$USER2_PUBKEY"

echo "Check privilege (admin)"
python bank-client.py --private-key "$ADMIN_PRIVKEY" check_privilege --account "$ADMIN_PUBKEY" 

echo "Check privilege (user1)"
python bank-client.py --private-key "$USER1_PRIVKEY" check_privilege --account "$USER1_PUBKEY" 

echo "Check privilege (user2)"
python bank-client.py --private-key "$USER2_PRIVKEY" check_privilege --account "$USER2_PUBKEY"